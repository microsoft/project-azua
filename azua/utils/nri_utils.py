import json
import os
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

from ..utils.io_utils import read_json_as, read_pickle, get_nth_parent_dir


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
    edges_train = np.reshape(edges_train, [-1, num_atoms ** 2])
    edges_train = np.array((edges_train + 1) / 2, dtype=np.int64)

    loc_valid = np.transpose(loc_valid, [0, 3, 1, 2])
    vel_valid = np.transpose(vel_valid, [0, 3, 1, 2])
    feat_valid = np.concatenate([loc_valid, vel_valid], axis=3)
    edges_valid = np.reshape(edges_valid, [-1, num_atoms ** 2])
    edges_valid = np.array((edges_valid + 1) / 2, dtype=np.int64)

    loc_test = np.transpose(loc_test, [0, 3, 1, 2])
    vel_test = np.transpose(vel_test, [0, 3, 1, 2])
    feat_test = np.concatenate([loc_test, vel_test], axis=3)
    edges_test = np.reshape(edges_test, [-1, num_atoms ** 2])
    edges_test = np.array((edges_test + 1) / 2, dtype=np.int64)

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train)
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)

    # Exclude self edges
    off_diag_idx = np.ravel_multi_index(
        np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)), [num_atoms, num_atoms],
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


def json2array(path):
    """
    The output of PC/GES in the causal-cmd package is a json file that contains the estimated edges.
    This function converts such output to a numpy array (adjacency matrix).
    If the term (i,j) is 1, it means that there is an edge from node j to node i. 
    If both (i,j) and (j,i) are 1, it means that the direction is not known (recall that the output of PC/GES is a cpdag).
    """
    with open(path, "r") as f:
        dic = json.load(f)
    num_nodes = len(dic["nodes"])
    output = np.zeros((num_nodes, num_nodes))
    dic_name2idx = {}
    for i in range(num_nodes):
        dic_name2idx["C{}".format(i + 1)] = i
    for edge in dic["edgesSet"]:
        if (edge["endpoint1"]["ordinal"] == 1) and (edge["endpoint2"]["ordinal"] == 1):
            raise ValueError("There is an edge with both endpoints set to 1")
        elif (edge["endpoint1"]["ordinal"] == 1) and (edge["endpoint2"]["ordinal"] == 0):
            from_idx = dic_name2idx[edge["node2"]["name"]]
            to_idx = dic_name2idx[edge["node1"]["name"]]
            output[to_idx, from_idx] = 1
        elif (edge["endpoint1"]["ordinal"] == 0) and (edge["endpoint2"]["ordinal"] == 1):
            from_idx = dic_name2idx[edge["node1"]["name"]]
            to_idx = dic_name2idx[edge["node2"]["name"]]
            output[to_idx, from_idx] = 1
        elif (edge["endpoint1"]["ordinal"] == 0) and (edge["endpoint2"]["ordinal"] == 0):
            idx1 = dic_name2idx[edge["node1"]["name"]]
            idx2 = dic_name2idx[edge["node2"]["name"]]
            output[idx1, idx2] = 1
            output[idx2, idx1] = 1
    return output


def is_there_adjacency(adj_matrix):
    """
    If input is (n,n), this returns a 1D array of size n*(n-1)/2 indicating whether each edge is present or not (not
    considering orientation).
    """
    mask = np.tri(adj_matrix.shape[0], k=-1, dtype=bool)
    is_there_forward = adj_matrix[mask].astype(bool)
    is_there_backward = (adj_matrix.T)[mask].astype(bool)
    return is_there_backward | is_there_forward


def get_adjacency_type(adj_matrix):
    """
    If input is (n,n), this returns a 1D array of size n*(n-1)/2 indicating the type of each edge (that is, 0 if
    there is no edge, 1 if it is forward, -1 if it is backward and 2 if it is in both directions or undirected). 
    """

    def aux(f, b):
        if f and b:
            return 2
        elif f and not b:
            return 1
        elif not f and b:
            return -1
        elif not f and not b:
            return 0

    mask = np.tri(adj_matrix.shape[0], k=-1, dtype=bool)
    is_there_forward = adj_matrix[mask].astype(bool)
    is_there_backward = (adj_matrix.T)[mask].astype(bool)
    out = np.array([aux(f, b) for (f, b) in zip(is_there_forward, is_there_backward)])
    return out


def is_there_edge(adj_matrix):
    """
    If input is (n,n), this returns a 1D array of size n*(n-1) indicating whether each edge is present or not (considering orientation).
    """
    mask = (np.ones_like(adj_matrix) - np.eye(adj_matrix.shape[0])).astype(bool)
    return adj_matrix[mask].astype(bool)


def edge_prediction_metrics_path(gt_path, pred_path):
    """
    Load the ground truth and the predicted adjacency matrices and computes the edge prediction metrics.
    """
    # Loading true and predicted adj matrices
    adj_matrix_true = np.loadtxt(gt_path, delimiter=",")
    adj_matrix_predicted = np.loadtxt(pred_path, delimiter=",")
    return edge_prediction_metrics(adj_matrix_true, adj_matrix_predicted)


def edge_prediction_metrics(adj_matrix_true, adj_matrix_predicted, adj_matrix_mask=None):
    """
    Computes the edge predicition metrics when the ground truth DAG (or CPDAG) is adj_matrix_true and the predicted one
    is adj_matrix_predicted. Both are numpy arrays.
    adj_matrix_mask is the mask matrix for adj_matrices, that indicates which subgraph is partially known in the ground
    truth. 0 indicates the edge is unknwon, and 1 indicates that the edge is known.
    """
    if adj_matrix_mask is None:
        adj_matrix_mask = np.ones_like(adj_matrix_true)

    assert ((adj_matrix_true == 0) | (adj_matrix_true == 1)).all()
    assert ((adj_matrix_predicted == 0) | (adj_matrix_predicted == 1)).all()
    results = {}

    # Computing adjacency precision/recall
    v_mask = is_there_adjacency(adj_matrix_mask)
    # v_mask is true only if we know about at least one direction of the edge
    v_true = is_there_adjacency(adj_matrix_true) & v_mask
    v_predicted = is_there_adjacency(adj_matrix_predicted) & v_mask
    recall = (v_true & v_predicted).sum() / (v_true.sum())
    precision = (v_true & v_predicted).sum() / (v_predicted.sum()) if v_predicted.sum() != 0 else 0.0
    fscore = 2 * recall * precision / (precision + recall) if (recall + precision) != 0 else 0.0
    results["adjacency_recall"] = recall
    results["adjacency_precision"] = precision
    results["adjacency_fscore"] = fscore

    # Computing orientation precision/recall
    v_mask = is_there_adjacency(adj_matrix_mask)
    v_true = get_adjacency_type(adj_matrix_true) * v_mask
    v_predicted = get_adjacency_type(adj_matrix_predicted) * v_mask
    recall = ((v_true == v_predicted) & (v_true != 0)).sum() / (v_true != 0).sum()
    precision = (
        ((v_true == v_predicted) & (v_predicted != 0)).sum() / (v_predicted != 0).sum()
        if (v_predicted != 0).sum() != 0
        else 0.0
    )
    fscore = 2 * recall * precision / (precision + recall) if (recall + precision) != 0 else 0.0
    results["orientation_recall"] = recall
    results["orientation_precision"] = precision
    results["orientation_fscore"] = fscore

    # Computing causal accuracy (as in https://github.com/TURuibo/Neuropathic-Pain-Diagnosis-Simulator/blob/master/source/CauAcc.py)
    v_mask = is_there_edge(adj_matrix_mask)
    # v_mask is true only if we know about the edge
    v_true = is_there_edge(adj_matrix_true) & v_mask
    v_predicted = is_there_edge(adj_matrix_predicted) & v_mask
    causal_acc = (v_true & v_predicted).sum() / v_true.sum()
    results["causal_accuracy"] = causal_acc

    # Compute SHD and number of nonzero edges
    results["shd"] = _shd(adj_matrix_true, adj_matrix_predicted)
    results["nnz"] = adj_matrix_predicted.sum()
    return results


def edge_prediction_metrics_multisample(
    adj_matrix_true, adj_matrices_predicted, adj_matrix_mask=None, compute_mean=True
):
    """
    Computes the edge predicition metrics when the ground truth DAG (or CPDAG) is adj_matrix_true and many predicted
    adjacencies are sampled from the distribution. Both are numpy arrays, adj_matrix_true has shape (n, n) and
    adj_matrices_predicted has shape (M, n, n), where M is the number of matrices sampled.
    """
    results = {}
    for i in range(adj_matrices_predicted.shape[0]):
        adj_matrix_predicted = adj_matrices_predicted[i, :, :]  # (n, n)
        results_local = edge_prediction_metrics(adj_matrix_true, adj_matrix_predicted, adj_matrix_mask=adj_matrix_mask)
        for k in results_local:
            if k not in results:
                results[k] = []
            results[k].append(results_local[k])

    if compute_mean:
        for k in results:
            results[k] = np.mean(results[k])
    return results


def edge_prediction_metrics_soft(gt_path, pred_path, thresholds):
    """
    Computes the edge prediction metrics for different thresholds over the predicted (soft) adjacency matrix. 
    """
    # Loading true and predicted adj matrices
    adj_matrix_true = np.loadtxt(gt_path, delimiter=",")
    adj_matrix_predicted = np.loadtxt(pred_path, delimiter=",")
    assert ((adj_matrix_true == 0) | (adj_matrix_true == 1)).all()
    assert ((adj_matrix_predicted >= 0) & (adj_matrix_predicted <= 1)).all()
    output = {
        "adjacency_recall": [],
        "adjacency_precision": [],
        "adjacency_fscore": [],
        "orientation_recall": [],
        "orientation_precision": [],
        "orientation_fscore": [],
        "causal_accuracy": [],
    }
    for thr in thresholds:
        adj_matrix_predicted_hard = (adj_matrix_predicted >= thr).astype(float)
        results = edge_prediction_metrics(adj_matrix_true, adj_matrix_predicted_hard)
        output["adjacency_recall"].append(results["adjacency"]["recall"])
        output["adjacency_precision"].append(results["adjacency"]["precision"])
        output["adjacency_fscore"].append(results["adjacency"]["fscore"])
        output["orientation_recall"].append(results["orientation"]["recall"])
        output["orientation_precision"].append(results["orientation"]["precision"])
        output["orientation_fscore"].append(results["orientation"]["fscore"])
        output["causal_accuracy"].append(results["causal_accuracy"])
    return output


def _shd(adj_true, adj_pred):
    """
    Computes Structural Hamming Distance as E+M+R, where E is the number of extra edges,
    M the number of missing edges, and R the number os reverse edges.
    """
    E, M, R = 0, 0, 0
    for i in range(adj_true.shape[0]):
        for j in range(adj_true.shape[0]):
            if j <= i:
                continue
            if adj_true[i, j] == 1 and adj_true[j, i] == 0:
                if adj_pred[i, j] == 0 and adj_pred[j, i] == 0:
                    M += 1
                elif adj_pred[i, j] == 0 and adj_pred[j, i] == 1:
                    R += 1
                elif adj_pred[i, j] == 1 and adj_pred[j, i] == 1:
                    E += 1
            if adj_true[i, j] == 0 and adj_true[j, i] == 1:
                if adj_pred[i, j] == 0 and adj_pred[j, i] == 0:
                    M += 1
                elif adj_pred[i, j] == 1 and adj_pred[j, i] == 0:
                    R += 1
                elif adj_pred[i, j] == 1 and adj_pred[j, i] == 1:
                    E += 1
            if adj_true[i, j] == 0 and adj_true[j, i] == 0:
                E += adj_pred[i, j] + adj_pred[j, i]
    return E + M + R


def piecewise_linear(x, start, width, max_val=1):
    """
    Piecewise linear function whose value is:
        0 if x<=start
        max_val if x>=start+width
        grows linearly from 0 to max_val if start<=x<=(start+width)
    It is used to define the coefficient of the DAG-loss in NRI-MV.
    """
    return max_val * max(min((x - start) / width, 1), 0)


def compute_penalty(A):
    """
    Computes the penalty associated to the adjacency matrix A for the Eedi subset experiment.
    A is required to be 9x9 and (i,j) is 0/1 (1 means edge from j-th node to i-th node).
    The three first nodes are group year 5, the next three are group year 8 and the next three are group year 11.
    An arrow from lower to higher group year is -1 (a good edge). The opposite is +1 (a bad edge). The rest of edges are 0. 
    """
    assert A.shape == (9, 9)
    assert ((A == 1) | (A == 0)).all()
    ones = np.ones((3, 3))
    zeros = np.zeros((3, 3))
    mask_good_edges = np.block([[zeros, zeros, zeros], [ones, zeros, zeros], [ones, ones, zeros]])
    mask_bad_edges = np.transpose(mask_good_edges)
    return np.sum(A * mask_bad_edges) - np.sum(A * mask_good_edges)


def compute_penalty_soft(A):
    """
    Computes the penalty given by function 'compute_penalty' for different thresholds over the (soft) matrix A.
    """
    assert A.shape == (9, 9)
    assert ((A <= 1) & (A >= 0)).all()
    penalties = []
    for thr in np.linspace(0, 1, 100)[::-1]:
        penalties.append(compute_penalty((A >= thr).astype(int)))
    return penalties


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


def compute_penalty_topics(A):
    """
    Computes the penalty associated to the adjacency matrix A for the Eedi topics experiment.
    There are eleven topics, each one is a group year.
    An arrow from lower to higher group year is -1 (a good edge). The opposite is +1 (a bad edge). 
    """
    mask_good_edges = np.tri(11, k=-1, dtype=bool)
    mask_bad_edges = mask_good_edges.transpose()
    count_good = np.sum(A[mask_good_edges])
    count_bad = np.sum(A[mask_bad_edges])
    return count_bad - count_good, count_good, count_bad


def compute_penalty_topics_soft(A):
    """
    Computes the penalty given by function 'compute_penalty_topics' for different thresholds over the (soft) matrix A.
    """
    penalty_list, count_good_list, count_bad_list = [], [], []
    for thr in np.linspace(0, 1, 100):
        penalty, count_good, count_bad = compute_penalty_topics((A >= thr).astype(int))
        penalty_list.append(penalty)
        count_good_list.append(count_good)
        count_bad_list.append(count_bad)
    return penalty_list, count_good_list, count_bad_list


def get_relationships_as_df(path, n=None, threshold=0.5):
    """
    Returns a DataFrame with the relationships in the (soft) adjacency matrix in path.
    If n is different from None, the 'n' strongest relationships are taken. Otherwise, the 'threshold' is used.
    """
    # Loading adjacency matrix
    A = np.loadtxt(path, delimiter=",")
    assert ((A >= 0) & (A <= 1)).all()

    # Selecting the relationships
    if n is None:
        to_indices, from_indices = np.where(A > threshold)
    else:
        to_indices, from_indices = np.unravel_index(np.argsort(A, axis=None)[::-1], A.shape)
        to_indices, from_indices = to_indices[0:n], from_indices[0:n]

    # Obatining the probs and ordering accordingly
    probs = A[to_indices, from_indices]
    ordered_idxs = np.argsort(probs)[::-1]
    probs, to_indices, from_indices = probs[ordered_idxs], to_indices[ordered_idxs], from_indices[ordered_idxs]

    # Converting to lists (probably not necessary, just to get rid of the numpy "layer")
    to_indices, from_indices, probs = to_indices.tolist(), from_indices.tolist(), probs.tolist()

    # Load needed variables
    ordered_nodes = read_json_as(
        os.path.join(get_nth_parent_dir(os.path.abspath(__file__), 2), "data/eedi_task_3_4_topics/ordered_nodes.json"), Dict[str, Any]
    )
    topic_to_name = read_pickle(
        os.path.join(get_nth_parent_dir(os.path.abspath(__file__), 2), "data/eedi_task_3_4_topics/topic_to_name.pkl")
    )
    topic_to_parent = read_pickle(
        os.path.join(get_nth_parent_dir(os.path.abspath(__file__), 2), "data/eedi_task_3_4_topics/topic_to_parent.pkl")
    )

    # Converting indices to names
    to_name_list = [
        f"{topic_to_name[ordered_nodes[to_idx]]} [{topic_to_name[topic_to_parent[ordered_nodes[to_idx]]]}] [{topic_to_name[topic_to_parent[topic_to_parent[ordered_nodes[to_idx]]]]}]"
        for to_idx in to_indices
    ]
    from_name_list = [
        f"{topic_to_name[ordered_nodes[from_idx]]} [{topic_to_name[topic_to_parent[ordered_nodes[from_idx]]]}] [{topic_to_name[topic_to_parent[topic_to_parent[ordered_nodes[from_idx]]]]}]"
        for from_idx in from_indices
    ]
    # Returning the desired dataframe
    return pd.DataFrame({"Prob": probs, "Topic 1": from_name_list, "Topic 2": to_name_list})


def get_adjacency_at_parent_level(path, threshold):
    """
    Returns the distribution of relationships across higher level topics.
    """
    # Loading ordered_nodes and the topic_to_parent dictionary
    ordered_nodes = read_json_as(
        os.path.join(get_nth_parent_dir(os.path.abspath(__file__), 2), "data/eedi_task_3_4_topics/ordered_nodes.json"), Dict[str, Any]
    )
    topic_to_name = read_pickle(
        os.path.join(get_nth_parent_dir(os.path.abspath(__file__), 2), "data/eedi_task_3_4_topics/topic_to_name.pkl")
    )
    topic_to_parent = read_pickle(
        os.path.join(get_nth_parent_dir(os.path.abspath(__file__), 2), "data/eedi_task_3_4_topics/topic_to_parent.pkl")
    )
    topics_level_2 = sorted(set([topic_to_parent[topic] for topic in ordered_nodes]))
    topics_level_2_name = [topic_to_name[topic] for topic in topics_level_2]
    topics_level_1 = sorted(set([topic_to_parent[topic] for topic in topics_level_2]))
    topics_level_1_name = [topic_to_name[topic] for topic in topics_level_1]

    # Obtaining the to/from indices for each edge
    to_indices, from_indices = np.where(np.loadtxt(path, delimiter=",") > threshold)

    # Creating the output matrices and filling them
    output_level_2 = np.zeros((len(topics_level_2), len(topics_level_2)))
    output_level_1 = np.zeros((len(topics_level_1), len(topics_level_1)))
    for from_index, to_index in zip(from_indices, to_indices):
        from_topic_level_2 = topic_to_parent[ordered_nodes[from_index]]
        to_topic_level_2 = topic_to_parent[ordered_nodes[to_index]]
        output_level_2[topics_level_2.index(to_topic_level_2), topics_level_2.index(from_topic_level_2)] += 1

        from_topic_level_1 = topic_to_parent[from_topic_level_2]
        to_topic_level_1 = topic_to_parent[to_topic_level_2]
        output_level_1[topics_level_1.index(to_topic_level_1), topics_level_1.index(from_topic_level_1)] += 1

    # Getting inside, perc, adj_matrix for each level
    inside_level_2 = np.sum(output_level_2[np.eye(output_level_2.shape[0], dtype=bool)])
    perc_level_2 = inside_level_2 / np.sum(output_level_2)
    inside_level_1 = np.sum(output_level_1[np.eye(output_level_1.shape[0], dtype=bool)])
    perc_level_1 = inside_level_1 / np.sum(output_level_1)
    adj_matrix_level_2 = pd.DataFrame(output_level_2, index=topics_level_2_name, columns=topics_level_2_name)
    adj_matrix_level_1 = pd.DataFrame(output_level_1, index=topics_level_1_name, columns=topics_level_1_name)

    return adj_matrix_level_2, adj_matrix_level_1, inside_level_2, perc_level_2, inside_level_1, perc_level_1


def get_intersection_and_union(folder, threshold=0.35):

    # Creating the matrices
    adj_matrix_sum_hard = np.zeros((57, 57))
    adj_matrix_sum_soft = np.zeros((57, 57))
    for i in range(5):
        adj_matrix_sum_hard += (
            np.loadtxt(os.path.join(folder, f"adj_matrix_predicted_{i}.csv"), delimiter=",") >= threshold
        ).astype(int)
        adj_matrix_sum_soft += np.loadtxt(os.path.join(folder, f"adj_matrix_predicted_{i}.csv"), delimiter=",")
    intersection = np.zeros_like(adj_matrix_sum_hard)
    union = np.zeros_like(adj_matrix_sum_hard)
    intersection[adj_matrix_sum_hard == 5] = adj_matrix_sum_soft[adj_matrix_sum_hard == 5] / 5
    union[adj_matrix_sum_hard >= 1] = adj_matrix_sum_soft[adj_matrix_sum_hard >= 1] / 5

    # Saving them
    save_folder = {
        "intersection": os.path.join(folder, "intersection", f"thr_{threshold}"),
        "union": os.path.join(folder, "union", f"thr_{threshold}"),
    }
    os.makedirs(save_folder["intersection"], exist_ok=True)
    os.makedirs(save_folder["union"], exist_ok=True)
    np.savetxt(os.path.join(save_folder["intersection"], "adj_matrix.csv"), intersection, delimiter=",")
    np.savetxt(os.path.join(save_folder["union"], "adj_matrix.csv"), union, delimiter=",")


def convert_temporal_adj_matrix_to_static(
    adj_matrix: np.ndarray, adj_matrix_2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """ This method converts two temporal adjacency matrices into large static graphs with compatible shapes for causual discovery evaluation.
    E.g. adj_matrix: [lag1,from,to] and adj_matrix2: [lag2,from,to] and lag1>lag2.
    They will be converted to large static graphs with shape [lag1*from,lag1*to], where we append 0 submatrix to adj_matrix2.

    Args:
        adj_matrix (np.ndarray): Adjacency matrix in the form of [lag, from, to] or [N, lag, from, to] where N is the number of sampled adjacency matrices.
        adj_matrix_2 (np.ndarray): Adjacency matrix in the form of [lag2, from, to] or [N, lag2, from, to] where N is the number of sampled adjacency matrices.
    """

    if len(adj_matrix.shape) == 3:
        adj_matrix = np.expand_dims(adj_matrix, axis=0)
    if len(adj_matrix_2.shape) == 3:
        adj_matrix_2 = np.expand_dims(adj_matrix_2, axis=0)

    n_nodes, n_lag_1 = adj_matrix.shape[2], adj_matrix.shape[1]
    n_nodes_2, n_lag_2 = adj_matrix_2.shape[2], adj_matrix_2.shape[1]
    assert (
        n_nodes == n_nodes_2
    ), f"The number of nodes for two input adjacency are not consistent. Adjacency matrix 1: {n_nodes} Adjacency matrix 2:{n_nodes_2}."

    # Adapt them to compatible shapes
    max_lag = max(n_lag_1, n_lag_2)
    adj_matrix = np.pad(adj_matrix, [(0, 0), (0, max_lag - n_lag_1), (0, 0), (0, 0)])
    adj_matrix_2 = np.pad(adj_matrix_2, [(0, 0), (0, max_lag - n_lag_2), (0, 0), (0, 0)])

    # Convert to static adjacency matrices
    static_graph_1 = convert_to_static(adj_matrix)
    static_graph_2 = convert_to_static(adj_matrix_2)

    return np.squeeze(static_graph_1), np.squeeze(static_graph_2)


def convert_to_static(adj_matrix: np.ndarray) -> np.ndarray:
    """
    This method converts the input temporal adjacency matrix into a larger static adjacency matrix.
    Args:
        adj_matrix (np.ndarray): Adjacency matrix with shape [N, lag, from, to] where N is the number of sampled adjacency matrices.

    Returns:
        static_adj_matrix (np.ndarray): Static adjacency matrix with shape [N, lag*from, lag*to].
    """

    n_nodes, n_lag = adj_matrix.shape[2], adj_matrix.shape[1]

    for n in range(len(adj_matrix)):
        # Concatenate to [lag*node,node]
        cur_adj_matrix = np.concatenate(adj_matrix[n], axis=0)
        # Zero submatrix
        zero_matrix = np.zeros((cur_adj_matrix.shape[0], n_nodes * (n_lag - 1)), dtype=bool)
        # Static graph
        cur_static_adj_matrix = np.expand_dims(np.concatenate((cur_adj_matrix, zero_matrix), axis=1), axis=0)
        if n == 0:
            static_adj_matrix = cur_static_adj_matrix
        else:
            static_adj_matrix = np.stack((static_adj_matrix, cur_static_adj_matrix))

    return static_adj_matrix
