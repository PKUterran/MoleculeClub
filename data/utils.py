import os
import numpy as np
import torch
from typing import Tuple, List
from copy import deepcopy
from rdkit.Chem import AllChem

LIST_SEED = [
    16880611,
    17760704,
    17890714,
    19491001,
    19900612,
]


def set_seed(seed: int, use_cuda=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)


def split_by_interval(n, i, given_list: list = None) -> List[List[int]]:
    temp = 0
    ret = []
    if given_list:
        while temp + i < n:
            ret.append([given_list[j] for j in range(temp, temp + i)])
            temp += i
        ret.append([given_list[j] for j in range(temp, n)])
    else:
        while temp + i < n:
            ret.append(list(range(temp, temp + i)))
            temp += i
        ret.append(list(range(temp, n)))
    return ret


def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def target_positions(mol) -> np.ndarray:
    return mol.GetConformer().GetPositions()


def rdkit_positions(mol, seed=0) -> np.ndarray:
    mol_ = deepcopy(mol)
    position = np.zeros([len(mol_.GetAtoms()), 3], np.float)
    try:
        AllChem.EmbedMolecule(mol_, randomSeed=seed)
        position = mol_.GetConformer().GetPositions()
    except ValueError:
        pass
    return position


def get_mean_std(array: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = torch.mean(array, dim=0, keepdim=True)
    std = torch.std(array - mean, dim=0, keepdim=True)
    return mean, std
