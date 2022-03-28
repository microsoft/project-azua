import numpy as np
import os
import graphviz


def make_coding_tensors(
    num_samples,
    num_variables,
    do_idx,
    do_value,
    reference_value=None,
    condition_idx=None,
    condition_value=None,
    target_idxs=None,
):

    intervention = np.full((num_samples, num_variables), np.nan)
    intervention[:, do_idx] = do_value
    reference = np.full((num_samples, num_variables), np.nan)

    if reference_value is not None:
        reference[:, do_idx] = reference_value

    conditioning = np.full((num_samples, num_variables), np.nan)
    if condition_idx is not None and condition_value is not None:
        conditioning[:, condition_idx] = condition_value

    targets = np.full((num_samples, num_variables), np.nan)
    if target_idxs is not None:
        for t in target_idxs:
            targets[:, t] = 1.0

    return conditioning, intervention, reference, targets


def extract_observations(sample_dict):
    """
    Extract observations from a sample dictionary into a 2d np array
    """
    done = False
    idx = 0
    samples = []
    while not done:
        if f"x{idx}" in sample_dict.keys():
            samples.append(sample_dict[f"x{idx}"])
            idx += 1
        else:
            done = True
    return np.stack(samples, axis=1)


def save_csvs(savedir, adjacency_matrix, data_all, train_data, test_data, metadata_matrix):
    np.savetxt(os.path.join(savedir, "adj_matrix.csv"), adjacency_matrix, delimiter=",", fmt="%i")
    np.savetxt(os.path.join(savedir, "all.csv"), data_all, delimiter=",")
    np.savetxt(os.path.join(savedir, "train.csv"), train_data, delimiter=",")
    np.savetxt(os.path.join(savedir, "test.csv"), test_data, delimiter=",")
    np.savetxt(os.path.join(savedir, "interventions.csv"), metadata_matrix, delimiter=",")


def finalise(savedir, train_data, adjacency_matrix, interventions, metadata):
    ##############################################################################
    # Create metadata
    ##############################################################################
    test_data = np.concatenate(interventions, axis=0)
    metadata = [np.concatenate(data, axis=0) for data in zip(*metadata)]
    metadata.append(test_data)
    metadata_matrix = np.concatenate(metadata, axis=1)

    data_all = np.concatenate([train_data, test_data], axis=0)

    ###############################################################################
    # Save CSV
    ###############################################################################
    save_csvs(savedir, adjacency_matrix, data_all, train_data, test_data, metadata_matrix)


def make_graph(adjacency_matrix, labels=None):
    idx = np.abs(adjacency_matrix) > 0.01
    dirs = np.where(idx)
    d = graphviz.Digraph(engine="dot")
    names = labels if labels else [f"x{i}" for i in range(len(adjacency_matrix))]
    for name in names:
        d.node(name)
    for to, from_, coef in zip(dirs[0], dirs[1], adjacency_matrix[idx]):
        d.edge(names[from_], names[to], label=str(coef))
    return d


def str_to_dot(string):
    """
    Converts input string from graphviz library to valid DOT graph format.
    """
    graph = string.replace("\n", ";").replace("\t", "")
    graph = graph[:9] + graph[10:-2] + graph[-1]  # Removing unnecessary characters from string
    return graph
