import torch
import numpy as np


def kabsch(pos: torch.Tensor, fit_pos: torch.Tensor, mol_node_matrix: torch.Tensor=None, use_cuda=False) \
        -> (torch.Tensor, torch.Tensor):
    if mol_node_matrix is None:
        mol_node_matrix = torch.FloatTensor(torch.ones([1, pos.shape[0]]))
    pos_list = []
    fit_pos_list = []
    for mask in mol_node_matrix:
        n = torch.sum(mask)
        p0 = pos[mask > 0, :]
        q0 = fit_pos[mask > 0, :]
        p = p0 - torch.sum(p0, dim=0) / n
        q = q0 - torch.sum(q0, dim=0) / n
        c = p.t() @ q
        det = torch.det(c)
        v, s, w = torch.svd(c)
        rd1 = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
        rd2 = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=torch.float32)
        if use_cuda:
            rd1 = rd1.cuda()
            rd2 = rd2.cuda()
        r1 = w @ rd1 @ v.t()
        r2 = w @ rd2 @ v.t()
        p1 = p @ r1
        p2 = p @ r2
        nd1 = torch.norm(p1 - q)
        nd2 = torch.norm(p2 - q)
        if det > 1e-5:
            pos_list.append(p1)
        elif det < -1e-5:
            pos_list.append(p2)
        else:
            if nd1 < nd2:
                pos_list.append(p1.detach())
            else:
                pos_list.append(p2.detach())
        fit_pos_list.append(q)

    ret_pos = torch.cat(pos_list, dim=0)
    ret_fit_pos = torch.cat(fit_pos_list, dim=0)
    return ret_pos, ret_fit_pos


def rmsd(src: torch.Tensor, tgt: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
    md2 = mass * torch.pow(src - tgt, 2).sum(dim=1, keepdim=True)
    loss = torch.sqrt(md2.sum() / mass.sum())
    return loss


if __name__ == '__main__':
    pos = torch.tensor([
        [1.2872716402317572, 0.10787202861021278, 0.0],
        [-0.09007753136792773, -0.40715148832140396, 0.0],
        [-1.1971941088638294, -0.8382876125923721, 0.0],
        [1.2872716402317572, 0.10787202861021278, 0.0],
        [-0.09007753136792773, -0.40715148832140396, 0.0],
        [-1.1971941088638294, -0.8382876125923721, 0.0],
        [0.7520094407284719, 0.0, 0.0],
        [-0.7520094407284719, 0.0, 0.0],
    ], dtype=torch.float32)
    fit_pos = torch.tensor([
        [-0.0178, 1.4644, 0.0101],
        [0.0021, 0.0095, 0.0020],
        [0.0183, -1.1918, -0.0045],
        [-1.2872716402317572, 0.10787202861021278, 0.0],
        [0.09007753136792773, -0.40715148832140396, 0.0],
        [1.1971941088638294, -0.8382876125923721, 0.0],
        [-0.0187, 1.5256, 0.0104],
        [0.0021, -0.0039, 0.0020],
    ], dtype=torch.float32)
    mnm = torch.tensor([
        # [1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1],
    ], dtype=torch.float32)
    pos, fit_pos = kabsch(pos, fit_pos, mnm)
    np.set_printoptions(precision=3, suppress=True)
    print(pos.numpy())
    print(fit_pos.numpy())
    r = rmsd(pos, fit_pos, torch.tensor([[1], [1], [1], [1], [1], [1], [1], [1]], dtype=torch.float32))
    print(r)
