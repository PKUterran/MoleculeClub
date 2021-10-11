import torch
import torch.nn as nn
from typing import Tuple, List, Dict, Any

from data.structures import MaskMatrices
from net.utils.components import MLP


class Force(nn.Module):
    ESP = 1e-6

    def __init__(self, v_dim: int, e_dim: int, q_dim: int, h_dim: int, param_dis=True):
        super(Force, self).__init__()
        self.param_dis = param_dis
        self.fb_linear = MLP(v_dim + e_dim, 1, hidden_dims=[h_dim] * 2, activation='tanh')

        if self.param_dis:
            self.fr_linear = MLP(v_dim + 1, 1, hidden_dims=[h_dim] * 2, activation='tanh', bias=False)
        else:
            self.fr_linear = MLP(1, 1, hidden_dims=[h_dim] * 2, activation='tanh', bias=False)

        nn.init.constant_(self.fb_linear.linears[-1].weight, 0.)
        nn.init.constant_(self.fb_linear.linears[-1].bias, 0.)
        nn.init.constant_(self.fr_linear.linears[-1].weight, 0.)
        self.fr_tanh = nn.Tanh()

    def forward(self, v: torch.Tensor, e: torch.Tensor, m: torch.Tensor, q: torch.Tensor,
                mask_matrices: MaskMatrices) -> torch.Tensor:
        # bond force
        vew1 = mask_matrices.vertex_edge_w1  # shape [n_vertex, n_edge]
        vew2 = mask_matrices.vertex_edge_w2  # shape [n_vertex, n_edge]
        vew_u = torch.cat([vew1, vew2], dim=1)  # shape [n_vertex, 2 * n_edge]
        vew_v = torch.cat([vew2, vew1], dim=1)  # shape [n_vertex, 2 * n_edge]
        e2 = torch.cat([e, e])  # shape [2 * n_edge, e_dim]
        uv_e = torch.cat([(vew_u + vew_v).t() @ v, e2], dim=1)
        delta_q = vew_u.t() @ q - vew_v.t() @ q
        unit_f_bond = delta_q / (torch.norm(delta_q, dim=1, keepdim=True) + self.ESP)
        value_f_bond = self.fb_linear(uv_e)
        f_bond = vew_u @ (unit_f_bond * value_f_bond)

        # relative force
        mvw = mask_matrices.mol_vertex_w
        vvm = mvw.t() @ mvw
        mm = m * m.reshape([1, -1])
        vv_massive_mask = vvm * mm

        delta_q = torch.unsqueeze(q, dim=1) - torch.unsqueeze(q, dim=0)
        delta_d = torch.abs(delta_q).norm(dim=2) + self.ESP

        unit_f_rela = delta_q / (torch.norm(delta_q, dim=2, keepdim=True) + self.ESP)
        if self.param_dis:
            add_v = torch.unsqueeze(v, dim=1) + torch.unsqueeze(v, dim=0)
            input_d = torch.cat([
                ((delta_d ** -2 - delta_d ** -1) * vv_massive_mask).unsqueeze(2),
                add_v * vv_massive_mask.unsqueeze(2)
            ], dim=2)
        else:
            input_d = ((delta_d ** -2 - delta_d ** -1) * vv_massive_mask).unsqueeze(2)
        value_f_rela = self.fr_linear(input_d)
        f_rela = (unit_f_rela * value_f_rela).sum(dim=1)

        f = f_bond + f_rela
        single_mask = vew_u.sum(dim=1) == 0
        f[single_mask] = f[single_mask].detach()
        return f


class NewtonianDerivation(nn.Module):
    def __init__(self, v_dim: int, e_dim: int, pq_dim: int, h_dim=256,
                 use_cuda=False, p_dropout=0.0):
        super(NewtonianDerivation, self).__init__()
        self.v_dim = v_dim
        self.e_dim = e_dim
        self.pq_dim = pq_dim
        self.use_cuda = use_cuda
        self.p_dropout = p_dropout
        self.force = Force(self.v_dim, self.e_dim, self.pq_dim, h_dim=h_dim)

    def forward(self, v: torch.Tensor, e: torch.Tensor, m: torch.Tensor, p: torch.Tensor, q: torch.Tensor,
                mask_matrices: MaskMatrices
                ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        v = torch.sigmoid(v)
        e = torch.sigmoid(e)
        # dq / dt = v = p / m
        dq = p / m

        # dp / dt = F
        dp = self.force(v, e, m, q, mask_matrices)

        return_dict = {}
        return dp, dq, return_dict
