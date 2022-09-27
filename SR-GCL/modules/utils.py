import copy

import torch
import numpy as np


def pad_batch(h_node, batch, max_input_len, get_mask=False):

    num_batch = batch[-1] + 1
    num_nodes = []
    masks = []
    for i in range(num_batch):
        mask = batch.eq(i)
        masks.append(mask)
        num_node = mask.sum()
        num_nodes.append(num_node)

    max_num_nodes = min(max(num_nodes), max_input_len)
    padded_h_node = h_node.data.new(max_num_nodes, num_batch, h_node.size(-1)).fill_(0)
    src_padding_mask = h_node.data.new(num_batch, max_num_nodes).fill_(0).bool()

    for i, mask in enumerate(masks):
        num_node = num_nodes[i]
        if num_node > max_num_nodes:
            num_node = max_num_nodes
        padded_h_node[-num_node:, i] = h_node[mask][-num_node:]
        src_padding_mask[i, : max_num_nodes - num_node] = True  # [b, s]

    if get_mask:
        return padded_h_node, src_padding_mask, num_nodes, masks, max_num_nodes

    return padded_h_node, src_padding_mask


def aug_batch(h_node, batch, attn_weights, aug_ratio, max_input_len, get_mask=False):
    num_batch = batch[-1] + 1
    num_nodes = []
    masks = []
    node_index_start = 0
    # node_index_end = 0
    node_mem = []
    for i in range(num_batch):
        mask = batch.eq(i)
        masks.append(mask)
        num_node = mask.sum().item()
        num_nodes.append(num_node)
        node_index_end = node_index_start + num_node
        rationale_node_num = int(num_node*aug_ratio)

        node_prob = attn_weights[node_index_start:node_index_end]
        node_prob += 0.001
        node_prob = np.array(node_prob)
        node_prob /= node_prob.sum()

        idx_nondrop = np.random.choice(num_node, rationale_node_num, replace=False, p=node_prob)

        idx_drop = np.setdiff1d(np.arange(num_node), idx_nondrop).tolist()
        # idx_drop.sort()

        idx_drop = [i+node_index_start for i in idx_drop]
        node_mem += idx_drop

        node_index_start = node_index_end

    shuffle_node = np.array(node_mem)
    np.random.shuffle(shuffle_node)
    ori_node = np.array(node_mem)

    h_aug_node = copy.deepcopy(h_node.detach())
    h_aug_node[ori_node] = h_aug_node[shuffle_node]

    max_num_nodes = min(max(num_nodes), max_input_len)
    padded_h_aug_node = h_aug_node.data.new(max_num_nodes, num_batch, h_aug_node.size(-1)).fill_(0)
    src_padding_mask = h_aug_node.data.new(num_batch, max_num_nodes).fill_(0).bool()

    for i, mask in enumerate(masks):
        num_node = num_nodes[i]
        if num_node > max_num_nodes:
            num_node = max_num_nodes
        padded_h_aug_node[-num_node:, i] = h_aug_node[mask][-num_node:]
        src_padding_mask[i, : max_num_nodes - num_node] = True  # [b, s]

    if get_mask:
        return padded_h_aug_node, src_padding_mask, num_nodes, masks, max_num_nodes
    return padded_h_aug_node, src_padding_mask


def unpad_batch(padded_h_node, prev_h_node, num_nodes, origin_mask, max_num_nodes):
    """
    padded_h_node: [s, b, f]
    prev_h_node: [bxs, f]
    batch: [n]
    pad_mask: [b, s]
    """

    for i, mask in enumerate(origin_mask):
        num_node = num_nodes[i]
        if num_node > max_num_nodes:
            num_node = max_num_nodes
            # cutoff mask
            indices = mask.nonzero()
            indices = indices[-num_node:]
            mask = torch.zeros_like(mask)
            mask[indices] = True
        # logger.info("prev_h_node:", prev_h_node.size())
        # logger.info("padded_h_node:", padded_h_node.size())
        # logger.info("mask:", mask.size())
        prev_h_node = prev_h_node.masked_scatter(mask.unsqueeze(-1), padded_h_node[-num_node:, i])
    return prev_h_node
