import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax, to_dense_batch
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from modules.gnn_module import GNNNodeEmbedding
from modules.masked_transformer_encoder import MaskedOnlyTransformerEncoder
from modules.transformer_encoder import TransformerNodeEncoder
from modules.utils import pad_batch, aug_batch

from loguru import logger
import math

from copy import deepcopy

num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3 

class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 
        

    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__()
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):

    def __init__(self, in_dim, out_dim, aggr="add"):
        super(GCNConv, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = torch.nn.Linear(in_dim, out_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, out_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, out_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype, device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        norm = self.norm(edge_index[0], x.size(0), x.dtype)

        x = self.linear(x)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr="add"):
        super(GATConv, self).__init__()

        self.aggr = aggr

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, heads * emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, heads * emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):

        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)
        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias

        return aggr_out


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr = "mean"):
        super(GraphSAGEConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        x = self.linear(x)

        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)


class GNN(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio=0, gnn_type = "gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        return node_representation


class GNN_graphpred(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, num_tasks, JK = "last", drop_ratio = 0, graph_pooling = "mean", gnn_type = "gin"):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type)

        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1
        
        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, self.num_tasks)

    def from_pretrained(self, model_file, device):
        if not model_file == "":
            self.gnn.load_state_dict(torch.load(model_file + "_gnn.pth", map_location=lambda storage, loc: storage))
            self.gnn.to(device)

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)

        return self.graph_pred_linear(self.pool(node_representation, batch))


class GraphTrans(torch.nn.Module):
    def __init__(self, args):
        super(GraphTrans, self).__init__()
        self.atom_encoder = AtomEncoder(emb_dim=args.gnn_emb_dim)
        self.bond_encoder = BondEncoder(emb_dim=args.gnn_emb_dim)
        self.gnn_node = GNNNodeEmbedding(
            args.gnn_virtual_node,
            args.gnn_num_layer,
            args.gnn_emb_dim,
            self.atom_encoder,
            self.bond_encoder,
            JK=args.gnn_JK,
            drop_ratio=args.gnn_dropout,
            residual=args.gnn_residual,
            gnn_type=args.gnn_type,
        )

        gnn_emb_dim = 2 * args.gnn_emb_dim if args.gnn_JK == "cat" else args.gnn_emb_dim
        self.gnn2transformer = nn.Linear(gnn_emb_dim, args.d_model)
        self.pos_encoder = PositionalEncoding(args.d_model, dropout=0) if args.pos_encoder else None
        self.transformer_encoder = TransformerNodeEncoder(args)
        self.masked_transformer_encoder = MaskedOnlyTransformerEncoder(args)
        self.num_encoder_layers = args.num_encoder_layers
        self.num_encoder_layers_masked = args.num_encoder_layers_masked

        self.pooling = args.graph_pooling
        self.d_model = args.d_model

    def forward(self, batched_data, perturb=None):
        h_node = self.gnn_node(batched_data, perturb)
        h_node = self.gnn2transformer(h_node)  # [s, b, d_model]

        padded_h_node, src_padding_mask = to_dense_batch(h_node, batched_data.batch)
        padded_h_node = padded_h_node.permute(1, 0, 2)
        max_num_nodes = padded_h_node.size(0)
        src_padding_mask = ~src_padding_mask

        # TODO(paras): implement mask
        transformer_out = padded_h_node
        if self.pos_encoder is not None:
            transformer_out = self.pos_encoder(transformer_out)
        if self.num_encoder_layers_masked > 0:
            adj_list = batched_data.adj_list
            padded_adj_list = torch.zeros((len(adj_list), max_num_nodes, max_num_nodes), device=h_node.device)
            for idx, adj_list_item in enumerate(adj_list):
                N, _ = adj_list_item.shape
                padded_adj_list[idx, 0:N, 0:N] = torch.from_numpy(adj_list_item)
            transformer_out = self.masked_transformer_encoder(
                transformer_out.transpose(0, 1), attn_mask=padded_adj_list, valid_input_mask=src_padding_mask
            ).transpose(0, 1)
        if self.num_encoder_layers > 0:
            transformer_out, _ = self.transformer_encoder(transformer_out, src_padding_mask)  # [s, b, h], [b, s]

        if self.pooling in ["last", "cls"]:
            h_graph = transformer_out[-1]
        elif self.pooling == "mean":
            h_graph = transformer_out.sum(0) / src_padding_mask.sum(-1, keepdim=True)
        else:
            raise NotImplementedError

        return h_graph

    def node_rep(self, batched_data, perturb=None):
        h_node = self.gnn_node(batched_data, perturb)
        h_node = self.gnn2transformer(h_node)  # [s, b, d_model]

        padded_h_node, src_padding_mask = to_dense_batch(h_node, batched_data.batch)
        padded_h_node = padded_h_node.permute(1, 0, 2)
        max_num_nodes = padded_h_node.size(0)
        src_padding_mask = ~src_padding_mask

        transformer_out = padded_h_node
        if self.pos_encoder is not None:
            transformer_out = self.pos_encoder(transformer_out)
        if self.num_encoder_layers_masked > 0:
            adj_list = batched_data.adj_list
            padded_adj_list = torch.zeros((len(adj_list), max_num_nodes, max_num_nodes), device=h_node.device)
            for idx, adj_list_item in enumerate(adj_list):
                N, _ = adj_list_item.shape
                padded_adj_list[idx, 0:N, 0:N] = torch.from_numpy(adj_list_item)
            transformer_out = self.masked_transformer_encoder(
                transformer_out.transpose(0, 1), attn_mask=padded_adj_list, valid_input_mask=src_padding_mask
            ).transpose(0, 1)
        if self.num_encoder_layers > 0:
            transformer_out, mask = self.transformer_encoder(transformer_out, src_padding_mask)  # [s, b, h], [b, s]

        node_rep = transformer_out[:-1,:,:].permute(1, 0, 2)
        node_rep = node_rep[~src_padding_mask, :]

        return node_rep

    @torch.no_grad()
    def attn_weights(self, batched_data, perturb=None):
        h_node = self.gnn_node(batched_data, perturb)
        h_node = self.gnn2transformer(h_node)  # [s, b, d_model]

        padded_h_node, src_padding_mask = to_dense_batch(h_node, batched_data.batch)
        padded_h_node = padded_h_node.permute(1, 0, 2)
        max_num_nodes = padded_h_node.size(0)
        src_padding_mask = ~src_padding_mask

        # TODO(paras): implement mask
        transformer_out = padded_h_node
        if self.pos_encoder is not None:
            transformer_out = self.pos_encoder(transformer_out)
        if self.num_encoder_layers_masked > 0:
            adj_list = batched_data.adj_list
            padded_adj_list = torch.zeros((len(adj_list), max_num_nodes, max_num_nodes), device=h_node.device)
            for idx, adj_list_item in enumerate(adj_list):
                N, _ = adj_list_item.shape
                padded_adj_list[idx, 0:N, 0:N] = torch.from_numpy(adj_list_item)
            transformer_out = self.masked_transformer_encoder(
                transformer_out.transpose(0, 1), attn_mask=padded_adj_list, valid_input_mask=src_padding_mask
            ).transpose(0, 1)
        if self.num_encoder_layers > 0:
            _, _, attn_weights = self.transformer_encoder.forward_attn(transformer_out, src_padding_mask)  # [s, b, h], [b, s]

        node_attn_weights = attn_weights[:, -1, :-1].contiguous().view(-1)[~src_padding_mask.view(-1)]
        edge_attention_weights = [attn_weights[i, :-1, :-1][batched_data.get_example(i).edge_index[0], batched_data.get_example(i).edge_index[1]] for i in range(batched_data.num_graphs)]
        edge_attention_weights = torch.cat(tuple(edge_attention_weights)).contiguous()
        return node_attn_weights, edge_attention_weights

    def epoch_callback(self, epoch):
        # TODO: maybe unfreeze the gnn at the end.
        if self.freeze_gnn is not None and epoch >= self.freeze_gnn:
            logger.info(f"Freeze GNN weight after epoch: {epoch}")
            for param in self.gnn_node.parameters():
                param.requires_grad = False

    def _gnn_node_state(self, state_dict):
        module_name = "gnn_node"
        new_state_dict = dict()
        for k, v in state_dict.items():
            if module_name in k:
                new_key = k.split(".")
                module_index = new_key.index(module_name)
                new_key = ".".join(new_key[module_index + 1:])
                new_state_dict[new_key] = v
        return new_state_dict


class GraphTrans_pred(torch.nn.Module):
    def __init__(self, args, num_tasks):
        super(GraphTrans_pred, self).__init__()
        self.atom_encoder = AtomEncoder(emb_dim=args.gnn_emb_dim)
        self.bond_encoder = BondEncoder(emb_dim=args.gnn_emb_dim)
        self.gnn_node = GNNNodeEmbedding(
            args.gnn_virtual_node,
            args.gnn_num_layer,
            args.gnn_emb_dim,
            self.atom_encoder,
            self.bond_encoder,
            JK=args.gnn_JK,
            drop_ratio=args.gnn_dropout,
            residual=args.gnn_residual,
            gnn_type=args.gnn_type,
        )

        gnn_emb_dim = 2 * args.gnn_emb_dim if args.gnn_JK == "cat" else args.gnn_emb_dim
        self.gnn2transformer = nn.Linear(gnn_emb_dim, args.d_model)
        self.pos_encoder = PositionalEncoding(args.d_model, dropout=0) if args.pos_encoder else None
        self.transformer_encoder = TransformerNodeEncoder(args)
        self.masked_transformer_encoder = MaskedOnlyTransformerEncoder(args)
        self.num_encoder_layers = args.num_encoder_layers
        self.num_encoder_layers_masked = args.num_encoder_layers_masked

        self.pooling = args.graph_pooling
        self.d_model = args.d_model

        self.num_tasks = num_tasks

        if not args.input_model_file == "":
            self.from_pretrained(args.input_model_file)

        self.graph_pred_linear_list = torch.nn.ModuleList()
        self.graph_pred_linear = torch.nn.Linear(args.d_model, self.num_tasks)

    def forward(self, batched_data, perturb=None):
        h_node = self.gnn_node(batched_data, perturb)
        h_node = self.gnn2transformer(h_node)  # [s, b, d_model]

        padded_h_node, src_padding_mask = to_dense_batch(h_node, batched_data.batch)
        padded_h_node = padded_h_node.permute(1, 0, 2)
        max_num_nodes = padded_h_node.size(0)
        src_padding_mask = ~src_padding_mask

        # TODO(paras): implement mask
        transformer_out = padded_h_node
        if self.pos_encoder is not None:
            transformer_out = self.pos_encoder(transformer_out)
        if self.num_encoder_layers_masked > 0:
            adj_list = batched_data.adj_list
            padded_adj_list = torch.zeros((len(adj_list), max_num_nodes, max_num_nodes), device=h_node.device)
            for idx, adj_list_item in enumerate(adj_list):
                N, _ = adj_list_item.shape
                padded_adj_list[idx, 0:N, 0:N] = torch.from_numpy(adj_list_item)
            transformer_out = self.masked_transformer_encoder(
                transformer_out.transpose(0, 1), attn_mask=padded_adj_list, valid_input_mask=src_padding_mask
            ).transpose(0, 1)
        if self.num_encoder_layers > 0:
            transformer_out, _ = self.transformer_encoder(transformer_out, src_padding_mask)  # [s, b, h], [b, s]

        if self.pooling in ["last", "cls"]:
            h_graph = transformer_out[-1]
        elif self.pooling == "mean":
            h_graph = transformer_out.sum(0) / src_padding_mask.sum(-1, keepdim=True)
        else:
            raise NotImplementedError

        return h_graph

    def pred(self, batched_data, perturb=None):
        h_graph = self.forward(batched_data)
        out = self.graph_pred_linear(h_graph)
        return out

    def epoch_callback(self, epoch):
        # TODO: maybe unfreeze the gnn at the end.
        if self.freeze_gnn is not None and epoch >= self.freeze_gnn:
            logger.info(f"Freeze GNN weight after epoch: {epoch}")
            for param in self.gnn_node.parameters():
                param.requires_grad = False

    def _gnn_node_state(self, state_dict):
        module_name = "gnn_node"
        new_state_dict = dict()
        for k, v in state_dict.items():
            if module_name in k:
                new_key = k.split(".")
                module_index = new_key.index(module_name)
                new_key = ".".join(new_key[module_index + 1:])
                new_state_dict[new_key] = v
        return new_state_dict

    def from_pretrained(self, path):
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        self.load_state_dict(state_dict)
        print("Loaded GraphTrans from " + path)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


if __name__ == "__main__":
    pass

