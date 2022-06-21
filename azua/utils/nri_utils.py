import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset


def load_particle_data(path, batch_size=1, suffix=""):
    loc_train = np.load(os.path.join(path, "loc_train" + suffix + ".npy"))
    vel_train = np.load(os.path.join(path, "vel_train" + suffix + ".npy"))
    edges_train = np.load(os.path.join(path, "edges_train" + suffix + ".npy"))

    loc_valid = np.load(os.path.join(path, "loc_valid" + suffix + ".npy"))
    vel_valid = np.load(os.path.join(path, "vel_valid" + suffix + ".npy"))
    edges_valid = np.load(os.path.join(path, "edges_valid" + suffix + ".npy"))

    loc_test = np.load(os.path.join(path, "loc_test" + suffix + ".npy"))
    vel_test = np.load(os.path.join(path, "vel_test" + suffix + ".npy"))
    edges_test = np.load(os.path.join(path, "edges_test" + suffix + ".npy"))

    # [num_samples, num_timesteps, num_dims, num_atoms]
    num_atoms = loc_train.shape[3]

    loc_max = loc_train.max()
    loc_min = loc_train.min()
    vel_max = vel_train.max()
    vel_min = vel_train.min()

    # Normalize to [-1, 1]
    loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_train = (vel_train - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_valid = (loc_valid - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_valid = (vel_valid - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_test = (loc_test - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_test = (vel_test - vel_min) * 2 / (vel_max - vel_min) - 1

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    loc_train = np.transpose(loc_train, [0, 3, 1, 2])
    vel_train = np.transpose(vel_train, [0, 3, 1, 2])
    feat_train = np.concatenate([loc_train, vel_train], axis=3)
    edges_train = np.reshape(edges_train, [-1, num_atoms**2])
    edges_train = np.array((edges_train + 1) / 2, dtype=np.int64)

    loc_valid = np.transpose(loc_valid, [0, 3, 1, 2])
    vel_valid = np.transpose(vel_valid, [0, 3, 1, 2])
    feat_valid = np.concatenate([loc_valid, vel_valid], axis=3)
    edges_valid = np.reshape(edges_valid, [-1, num_atoms**2])
    edges_valid = np.array((edges_valid + 1) / 2, dtype=np.int64)

    loc_test = np.transpose(loc_test, [0, 3, 1, 2])
    vel_test = np.transpose(vel_test, [0, 3, 1, 2])
    feat_test = np.concatenate([loc_test, vel_test], axis=3)
    edges_test = np.reshape(edges_test, [-1, num_atoms**2])
    edges_test = np.array((edges_test + 1) / 2, dtype=np.int64)

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train)
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)

    # Exclude self edges
    off_diag_idx = np.ravel_multi_index(
        np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
        [num_atoms, num_atoms],
    )
    edges_train = edges_train[:, off_diag_idx]  # shape: [num_sims, num_atoms*(num_atoms-1)]
    edges_valid = edges_valid[:, off_diag_idx]
    edges_test = edges_test[:, off_diag_idx]

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader


def kl_categorical(preds, log_prior, num_atoms, eps=1e-16):
    """
    preds: [num_sims, num_edges, num_edge_types]
    log_prior: [1, 1, num_edge_types]
    """
    kl_div = preds * (torch.log(preds + eps) - log_prior)
    return kl_div.sum() / (num_atoms * preds.size(0))


def kl_categorical_uniform(preds, num_atoms, num_edge_types, add_const=False, eps=1e-16):
    kl_div = preds * torch.log(preds + eps)
    if add_const:
        const = np.log(num_edge_types)
        kl_div += const
    return kl_div.sum() / (num_atoms * preds.size(0))


def nll_gaussian(preds, target, variance, add_const=False):
    neg_log_p = (preds - target) ** 2 / (2 * variance)
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0) * target.size(1))


def edge_accuracy(preds, target):
    _, preds = preds.max(-1)
    correct = preds.float().data.eq(target.float().data.view_as(preds)).cpu().sum()
    return np.float(correct) / (target.size(0) * target.size(1))


def piecewise_linear(x, start, width, max_val=1):
    """
    Piecewise linear function whose value is:
        0 if x<=start
        max_val if x>=start+width
        grows linearly from 0 to max_val if start<=x<=(start+width)
    It is used to define the coefficient of the DAG-loss in NRI-MV.
    """
    return max_val * max(min((x - start) / width, 1), 0)


def compute_dag_loss(vec, num_nodes):
    """
    vec is a n*(n-1) array with the flattened adjacency matrix (without the diag).
    """
    dev = vec.device
    adj_matrix = torch.zeros(num_nodes, num_nodes, device=dev)
    mask = (torch.ones(num_nodes, num_nodes, device=dev) - torch.eye(num_nodes, device=dev)).to(bool)
    adj_matrix[mask] = vec
    return torch.abs(torch.trace(torch.matrix_exp(adj_matrix * adj_matrix)) - num_nodes)


def get_feature_indices_per_node(variables):
    """
    Returns a list in which the i-th element is a list with the features indices that correspond to the i-th node.
    For each Variable in 'variables' argument, the node is specified through the group_name field.
    """
    nodes = [v.group_name for v in variables]
    nodes_unique = sorted(set(nodes))
    if len(nodes_unique) == len(nodes):
        nodes_unique = nodes
    output = []
    for node in nodes_unique:
        output.append([i for (i, e) in enumerate(nodes) if e == node])
    return output, nodes_unique
