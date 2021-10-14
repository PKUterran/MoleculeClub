import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
from sklearn.metrics import roc_auc_score

from data.structures import MaskMatrices
from net.utils import normalize_adj_rc, nonzero
from .kabsch import kabsch


def multi_roc(source: List[np.ndarray], target: np.ndarray) -> Tuple[float, List[float]]:
    list_roc = []
    n_m = len(source)
    for i in range(n_m):
        target_i = target[:, i]
        target_i = target_i[np.logical_not(np.isnan(target_i))]
        src = torch.softmax(torch.from_numpy(source[i]), dim=-1).numpy()
        tgt = np.zeros_like(src)
        for j in range(target_i.shape[0]):
            tgt[j, int(target_i[j])] = 1.
        try:
            roc = roc_auc_score(tgt, src)
        except ValueError:
            roc = 1
        list_roc.append(roc)
    return sum(list_roc) / len(list_roc), list_roc


def multi_mse_loss(source: torch.Tensor, target: torch.Tensor, explicit=False) -> torch.Tensor:
    se = (source - target) ** 2
    mse = torch.mean(se, dim=0)
    if explicit:
        return mse
    else:
        return torch.sum(mse)


def multi_mae_loss(source: torch.Tensor, target: torch.Tensor, explicit=False) -> torch.Tensor:
    ae = torch.abs(source - target)
    mae = torch.mean(ae, dim=0)
    if explicit:
        return mae
    else:
        return torch.sum(mae)


def mse_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(source, target)


def rmse_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(source, target).sqrt()


def mae_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(source - target))


def distance_among(positions: torch.Tensor) -> torch.Tensor:
    p1 = torch.unsqueeze(positions, 0)
    p2 = torch.unsqueeze(positions, 1)
    distance = torch.norm(p1 - p2, dim=2)
    return distance


def generate_adj(mask_matrices: MaskMatrices, mode, use_cuda=False) -> torch.Tensor:
    if mode == 'adj2':
        vew1 = mask_matrices.vertex_edge_w1
        vew2 = mask_matrices.vertex_edge_w2
        adj_d = vew1 @ vew2.t()
        i = torch.eye(adj_d.shape[0])
        if use_cuda:
            i = i.cuda()
        adj = adj_d + adj_d.t() + i
        adj_2 = adj @ adj
        mean_adj_2 = normalize_adj_rc(nonzero(adj_2))
        return mean_adj_2
    elif mode == 'adj3':
        vew1 = mask_matrices.vertex_edge_w1
        vew2 = mask_matrices.vertex_edge_w2
        adj_d = vew1 @ vew2.t()
        i = torch.eye(adj_d.shape[0])
        if use_cuda:
            i = i.cuda()
        adj = adj_d + adj_d.t() + i
        adj_2 = adj @ adj
        adj_3 = adj_2 @ adj
        mean_adj_3 = normalize_adj_rc(nonzero(adj_3))
        return mean_adj_3
    elif mode == 'adj4':
        vew1 = mask_matrices.vertex_edge_w1
        vew2 = mask_matrices.vertex_edge_w2
        adj_d = vew1 @ vew2.t()
        i = torch.eye(adj_d.shape[0])
        if use_cuda:
            i = i.cuda()
        adj = adj_d + adj_d.t() + i
        adj_2 = adj @ adj
        adj_4 = adj_2 @ adj_2
        mean_adj_3 = normalize_adj_rc(nonzero(adj_4))
        return mean_adj_3
    elif mode == 'norm_adj3':
        vew1 = mask_matrices.vertex_edge_w1
        vew2 = mask_matrices.vertex_edge_w2
        adj_d = vew1 @ vew2.t()
        i = torch.eye(adj_d.shape[0])
        if use_cuda:
            i = i.cuda()
        adj = adj_d + adj_d.t() + i
        norm_adj = normalize_adj_rc(adj)
        norm_adj_2 = norm_adj @ norm_adj
        norm_adj_3 = norm_adj_2 @ norm_adj
        mean_adj_3 = (norm_adj + norm_adj_2 + norm_adj_3) / 3
        return mean_adj_3
    elif mode == 'distance':
        n_mol = mask_matrices.mol_vertex_w.shape[0]
        mvw = mask_matrices.mol_vertex_w
        vv = mvw.t() @ mvw
        norm_vv = vv / ((torch.sum(vv, dim=1) ** 2) * n_mol)
        return norm_vv
    else:
        assert False, f'{mode}'


def adj3_loss(source: torch.Tensor, target: torch.Tensor, mask_matrices: MaskMatrices,
              use_cuda=False) -> torch.Tensor:
    n_atom = mask_matrices.mol_vertex_w.shape[1]
    mean_adj_3 = generate_adj(mask_matrices, mode='adj3', use_cuda=use_cuda)

    ds = distance_among(source)
    dt = distance_among(target)
    distance_2 = (ds - dt) ** 2
    loss = torch.sum(distance_2 * mean_adj_3) / n_atom
    return loss


def hierarchical_adj2_loss(sources: List[torch.Tensor], target: torch.Tensor, mask_matrices: MaskMatrices, weight=1.6,
                           use_cuda=False) -> torch.Tensor:
    n_s = len(sources)
    n_atom = mask_matrices.mol_vertex_w.shape[1]
    mean_adj_2 = generate_adj(mask_matrices, mode='adj2', use_cuda=use_cuda)

    w, t = [], 1
    for i in range(n_s):
        w.append(t)
        t *= weight
    tw = sum(w)
    w = [j / tw for j in w]

    losses = []
    for i in range(n_s):
        ds = distance_among(sources[i])
        dt = distance_among(target)
        distance_2 = (ds - dt) ** 2
        loss = torch.sum(distance_2 * mean_adj_2) * w[i] / n_atom
        losses.append(loss)

    return sum(losses)


def hierarchical_adj3_loss(sources: List[torch.Tensor], target: torch.Tensor, mask_matrices: MaskMatrices, weight=1.6,
                           use_cuda=False) -> torch.Tensor:
    n_s = len(sources)
    n_atom = mask_matrices.mol_vertex_w.shape[1]
    mean_adj_3 = generate_adj(mask_matrices, mode='adj3', use_cuda=use_cuda)

    w, t = [], 1
    for i in range(n_s):
        w.append(t)
        t *= weight
    tw = sum(w)
    w = [j / tw for j in w]

    losses = []
    for i in range(n_s):
        ds = distance_among(sources[i])
        dt = distance_among(target)
        distance_2 = (ds - dt) ** 2
        loss = torch.sum(distance_2 * mean_adj_3) * w[i] / n_atom
        losses.append(loss)

    return sum(losses)


def hierarchical_adj4_loss(sources: List[torch.Tensor], target: torch.Tensor, mask_matrices: MaskMatrices, weight=1.6,
                           use_cuda=False) -> torch.Tensor:
    n_s = len(sources)
    n_atom = mask_matrices.mol_vertex_w.shape[1]
    mean_adj_4 = generate_adj(mask_matrices, mode='adj4', use_cuda=use_cuda)

    w, t = [], 1
    for i in range(n_s):
        w.append(t)
        t *= weight
    tw = sum(w)
    w = [j / tw for j in w]

    losses = []
    for i in range(n_s):
        ds = distance_among(sources[i])
        dt = distance_among(target)
        distance_2 = (ds - dt) ** 2
        loss = torch.sum(distance_2 * mean_adj_4) * w[i] / n_atom
        losses.append(loss)

    return sum(losses)


def distance_loss(source: torch.Tensor, target: torch.Tensor, mask_matrices: MaskMatrices,
                  use_cuda=False, root_square=True) -> torch.Tensor:
    norm_vv = generate_adj(mask_matrices, mode='distance', use_cuda=use_cuda)
    ds = distance_among(source)
    dt = distance_among(target)
    if root_square:
        return torch.sqrt(torch.sum(((ds - dt) ** 2) * norm_vv))
    else:
        return torch.sum(torch.abs(ds - dt) * norm_vv)


def kabsch_rmsd_loss(source: torch.Tensor, target: torch.Tensor, mask_matrices: MaskMatrices,
                     use_cuda=False) -> torch.Tensor:
    pos, fit_pos = kabsch(source, target, mask_matrices.mol_vertex_w, use_cuda=use_cuda)
    return rmse_loss(pos, fit_pos)


def hierarchical_mixed_kabsch_adj3_loss(sources: List[torch.Tensor], target: torch.Tensor, mask_matrices: MaskMatrices,
                                        use_cuda=False) -> torch.Tensor:
    kabsch_rmsd = kabsch_rmsd_loss(sources[-1], target, mask_matrices, use_cuda=use_cuda)
    h_adj3 = hierarchical_adj3_loss(sources, target, mask_matrices, use_cuda=use_cuda)
    return 0.1 * kabsch_rmsd + h_adj3
