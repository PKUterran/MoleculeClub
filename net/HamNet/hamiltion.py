import torch
import torch.nn as nn
import torch.autograd as autograd
from typing import Union, Tuple

from data.structures import MaskMatrices


class KineticEnergy(nn.Module):
    def __init__(self, v_dim, p_dim, h_dim=16, dropout=0.0):
        super(KineticEnergy, self).__init__()
        self.W = nn.Linear(v_dim + p_dim, h_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.softplus = nn.Softplus()

    def forward(self, v, p, m):
        alpha = 1 / m
        vp = torch.cat([v, p], dim=1)
        pw = self.W(vp)
        pw = self.dropout(self.softplus(pw))
        apwwp = alpha * (pw ** 2)
        if torch.isnan(apwwp.sum()):
            apwwp[torch.isnan(apwwp)] = 0
        t = torch.sum(apwwp, dim=1, keepdim=True)
        return t


class PotentialEnergy(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim=16, dropout=0.0, use_cuda=False):
        super(PotentialEnergy, self).__init__()
        self.use_cuda = use_cuda
        self.linear1 = nn.Linear(v_dim + q_dim, h_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.softplus = nn.Softplus()

    def forward(self, v, q, m, vvm):
        norm_m = m
        mm = norm_m * norm_m.reshape([1, -1])
        eye = torch.eye(vvm.shape[1], dtype=torch.float32)
        if self.use_cuda:
            eye = eye.cuda()
        mask = vvm * mm
        vq = torch.cat([v, q], dim=1)
        delta_vq = torch.unsqueeze(vq, dim=0) - torch.unsqueeze(vq, dim=1)
        root = self.linear1(delta_vq)
        root = self.dropout(root)
        distance = (self.softplus(torch.sum(root ** 2, dim=2))) * (-eye + 1) + eye
        energy = mask * (distance ** -2 - distance ** -1)
        if torch.isnan(energy.sum()):
            energy[torch.isnan(energy)] = 0
        p = torch.sum(energy, dim=1, keepdim=True)
        return p


class DissipatedEnergy(nn.Module):
    def __init__(self, p_dim, h_dim=16, dropout=0.0):
        super(DissipatedEnergy, self).__init__()
        self.W = nn.Linear(p_dim, h_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.softplus = nn.Softplus()

    def forward(self, p, m):
        alpha2 = 1 / (m ** 2)
        pw = self.W(p)
        pw = self.dropout(self.softplus(pw))
        a2pwwp = alpha2 * (pw ** 2)
        if torch.isnan(a2pwwp.sum()):
            a2pwwp[torch.isnan(a2pwwp)] = 0
        f = torch.sum(a2pwwp, dim=1, keepdim=True)
        return f


class DissipativeHamiltonianDerivation(nn.Module):
    def __init__(self, v_dim: int, e_dim: int, p_dim: int, q_dim: int,
                 use_cuda=False, dropout=0.0):
        super(DissipativeHamiltonianDerivation, self).__init__()
        self.T = KineticEnergy(v_dim, p_dim)
        self.U = PotentialEnergy(v_dim, q_dim, dropout=dropout, use_cuda=use_cuda)
        self.F = DissipatedEnergy(p_dim)

    def forward(self, v: torch.Tensor, e: torch.Tensor, m: torch.Tensor, p: torch.Tensor, q: torch.Tensor,
                mask_matrices: MaskMatrices,
                return_energy=False, dissipate=True
                ) -> Union[Tuple[torch.Tensor, torch.Tensor],
                           Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        mvw = mask_matrices.mol_vertex_w
        vvm = mvw.t() @ mvw
        v, e = torch.sigmoid(v), torch.sigmoid(e)
        hamiltonians = self.T(v, p, m) + self.U(v, q, m, vvm)
        dissipations = self.F(p, m)
        hamilton = hamiltonians.sum()
        dissipated = dissipations.sum()
        dq = autograd.grad(hamilton, p, create_graph=True)[0]
        if dissipate:
            dp = -1 * (autograd.grad(hamilton, q, create_graph=True)[0] +
                       autograd.grad(dissipated, p, create_graph=True)[0] * m)
        else:
            dp = -1 * autograd.grad(hamilton, q, create_graph=True)[0]
        if return_energy:
            return dp, dq, hamiltonians, dissipations
        return dp, dq
