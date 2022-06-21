"""
Helper functions.
"""
import os
import sys
from contextlib import contextmanager
from typing import Tuple, overload

import git
import numpy as np
import torch


@overload
def to_tensors(
    array1: np.ndarray,
    array2: np.ndarray,
    device: torch.device,
    dtype: torch.dtype = torch.float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ...


@overload
def to_tensors(*arrays: np.ndarray, device: torch.device, dtype: torch.dtype = torch.float) -> Tuple[torch.Tensor, ...]:
    ...


def to_tensors(*arrays, device, dtype=torch.float):
    return tuple(torch.as_tensor(array, dtype=dtype, device=device) for array in arrays)


@contextmanager
def maintain_random_state(do_maintain=True):
    torch_rand_state = torch.get_rng_state()
    np_rand_state = np.random.get_state()
    if torch.cuda.is_available():
        cuda_rand_state = torch.cuda.get_rng_state()
    else:
        cuda_rand_state = None

    try:
        yield (torch_rand_state, np_rand_state, cuda_rand_state)
    finally:
        if do_maintain:
            if torch_rand_state is not None:
                torch.set_rng_state(torch_rand_state)
            if cuda_rand_state is not None:
                torch.cuda.set_rng_state(cuda_rand_state)
            if np_rand_state is not None:
                np.random.set_state(np_rand_state)


def get_random_state():
    """
    Get random states for PyTorch, PyTorch CUDA and Numpy.

    Returns:
        Dictionary of state type: state value.
    """
    states = {
        "torch_rand_state": torch.get_rng_state(),
        "np_rand_state": np.random.get_state(),
    }

    if torch.cuda.is_available():
        states["cuda_rand_state"] = torch.cuda.get_rng_state()

    return states


def write_git_info(directory: str, exist_ok: bool = False):
    """
    Write sys.argv, git hash, git diff to <directory>/git_info.txt

    directory: where to write git_info.txt.  This directory must already exist
    exist_ok: if set to True, may silently overwrite old git info
    """
    assert os.path.exists(directory)
    try:
        repo = git.Repo(search_parent_directories=True)

    except git.InvalidGitRepositoryError:
        # Likely to happen if we are in an AzureML run.
        raise ValueError("Not running inside a Git repo.")
    commit = repo.head.commit
    diff = repo.git.diff(None)
    mode = "w" if exist_ok else "x"
    with open(os.path.join(directory, "git_info.txt"), mode, encoding="utf-8") as f:
        f.write(f"sys.argv: {sys.argv}\n")
        f.write("Git commit: " + str(commit) + "\n")
        try:
            f.write("Active branch: " + str(repo.active_branch) + "\n")
        except TypeError:
            # Happens in PR build, detached head state
            pass
        f.write("Git diff:\n" + str(diff))
