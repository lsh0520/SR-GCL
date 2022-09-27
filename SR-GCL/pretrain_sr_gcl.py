import argparse
import time

from loader import MoleculeDatasetTransAug
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from model import GraphTrans


from copy import deepcopy
import gc


class graphcl(nn.Module):
    def __init__(self, graphtrans):
        super(graphcl, self).__init__()
        self.graphtrans = graphtrans
        self.d_model = self.graphtrans.d_model
        self.projection_head = nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.ReLU(inplace=True), nn.Linear(self.d_model, self.d_model))

    def forward_cl(self, batch_data):
        x = self.graphtrans(batch_data)
        x = self.projection_head(x)
        return x

    def attn_weights(self, batch_data):
        attn_weights = self.graphtrans.attn_weights(batch_data)
        return attn_weights

    def loss_cl(self, x1, x2, args):
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / args.temp)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        if args.loss_type == "nt":
            loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        elif args.loss_type == "nce":
            loss = pos_sim / sim_matrix.sum(dim=1)
        else:
            print('Undefined loss type')
            assert False
        loss = - torch.log(loss).mean()
        return loss


def train(args, epoch, model, device, dataset, optimizer):
    dataset.aug = "none"
    if epoch % args.attn_freq == 0:
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        model.eval()
        torch.set_grad_enabled(False)
        for step, batch in enumerate(loader):
            index_start = step * args.batch_size
            index_end = min(index_start + args.batch_size, len(dataset))
            batch = batch.to(device)
            node_attn_weights, edge_attn_weights = model.attn_weights(batch)
            dataset.node_score[dataset.slices['x'][index_start]:dataset.slices['x'][index_end]] = torch.squeeze(node_attn_weights.half())
            dataset.edge_score[dataset.slices['edge_index'][index_start]:dataset.slices['edge_index'][index_end]] = torch.squeeze(edge_attn_weights.half())

    dataset1 = deepcopy(dataset)
    dataset1 = dataset1.shuffle()
    dataset2 = deepcopy(dataset1)

    dataset1.aug, dataset1.aug_ratio = args.aug1, args.aug_ratio1
    dataset2.aug, dataset2.aug_ratio = args.aug2, args.aug_ratio2

    loader1 = DataLoader(dataset1, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    loader2 = DataLoader(dataset2, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    torch.set_grad_enabled(True)
    model.train()

    train_loss_accum = 0

    for step, batch in enumerate(zip(loader1, loader2)):
        batch1, batch2 = batch
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)

        optimizer.zero_grad()

        x1 = model.forward_cl(batch1)
        x2 = model.forward_cl(batch2)
        loss = model.loss_cl(x1, x2, args)

        loss.backward()
        optimizer.step()

        train_loss_accum += float(loss.detach().cpu().item())

    del dataset1, dataset2
    gc.collect()
    return train_loss_accum/(step+1)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')

    parser.add_argument('--gnn_type', type=str, default='gin')
    parser.add_argument('--gnn_emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--gnn_virtual_node', action='store_true', default=False)
    parser.add_argument('--gnn_dropout', type=float, default=0.3)
    parser.add_argument('--gnn_num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--gnn_JK', type=str, default='cat')
    parser.add_argument('--gnn_residual', action='store_true', default=False)
    parser.add_argument('--graph_pooling', type=str, default='cls')

    parser.add_argument('--pos_encoder', action='store_true', default=False)

    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4, help="transformer heads")
    parser.add_argument("--dim_feedforward", type=int, default=512, help="transformer feedforward dim")
    parser.add_argument("--transformer_dropout", type=float, default=0.3)
    parser.add_argument("--transformer_activation", type=str, default="relu")
    parser.add_argument("--num_encoder_layers", type=int, default=4)
    parser.add_argument("--max_input_len", default=1000, help="The max input length of transformer input")
    parser.add_argument("--transformer_norm_input", action="store_true", default=True)

    parser.add_argument("--num_encoder_layers_masked", type=int, default=0)
    parser.add_argument("--transformer_prenorm", action="store_true", default=False)

    parser.add_argument('--dataset', type=str, default='zinc_standard_agent',
                        help='root directory of dataset. For now, only classification.')
    parser.add_argument('--output_model_file', type=str, default='', help='filename to output the pre-trained model')
    parser.add_argument('--seed', type=int, default=12344, help="Ranom Seed")
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataset loading')
    parser.add_argument('--aug1', type=str, default='dropN_attn')
    parser.add_argument('--aug_ratio1', type=float, default=0.2)
    parser.add_argument('--aug2', type=str, default='dropE_attn')
    parser.add_argument('--aug_ratio2', type=float, default=0.3)

    parser.add_argument('--attn_freq', type=int, default=5)
    parser.add_argument('--temp', type=float, default=0.1)
    parser.add_argument('--loss_type', type=str, default='nce',
                        help='Contrastive loss to use [nt|nce]')
    args = parser.parse_args()

    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


    #set up dataset
    dataset = MoleculeDatasetTransAug("dataset/" + args.dataset, dataset=args.dataset)
    print(dataset)

    #set up model
    gnn = GraphTrans(args)

    model = graphcl(gnn)

    model.to(device)

    #set up optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay)

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))

        tic = time.time()
        train_loss = train(args, epoch, model, device, dataset, optimizer)
        toc = time.time()
        print(f"Time cost: {toc - tic} seconds")
        print(train_loss)

        if epoch % 20 == 0:
            save_path = f"./models_graphtrans/graphtrans_srgcl-epoch{epoch}.pth"
            torch.save(gnn.state_dict(), save_path)


if __name__ == "__main__":
    main()
