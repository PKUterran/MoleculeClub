import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Any

from net.utils import activation_select


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int,
                 hidden_dims: list = None, layer_activation: str = 'leaky_relu', activation: str = 'no',
                 use_cuda=False, bias=True, residual=False, dropout=0.0):
        super(MLP, self).__init__()
        self.use_cuda = use_cuda
        self.residual = residual
        self.dropout = nn.Dropout(p=dropout)

        if not hidden_dims:
            hidden_dims = []
        in_dims = [in_dim] + hidden_dims
        out_dims = hidden_dims + [out_dim]
        self.linears = nn.ModuleList([nn.Linear(i, o, bias=bias) for i, o in zip(in_dims, out_dims)])
        self.layer_act = activation_select(layer_activation)
        self.activate = activation_select(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, linear in enumerate(self.linears):
            x2 = linear(self.dropout(x))
            if i < len(self.linears) - 1:
                x2 = self.layer_act(x2)
            if self.residual:
                x = torch.cat([x, x2])
            else:
                x = x2
        x = self.activate(x)
        return x


class GCN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dims: list = None, activation: str = 'no',
                 use_cuda=False, residual=False):
        super(GCN, self).__init__()
        self.use_cuda = use_cuda
        self.residual = residual

        if not hidden_dims:
            hidden_dims = []
        in_dims = [in_dim] + hidden_dims
        out_dims = hidden_dims + [out_dim]
        self.linears = nn.ModuleList([nn.Linear(i, o, bias=True) for i, o in zip(in_dims, out_dims)])
        self.layer_act = nn.LeakyReLU()
        self.activate = activation_select(activation)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == a.shape[0]
        xs = [x]
        for i, linear in enumerate(self.linears):
            x2 = a @ linear(x)
            if i < len(self.linears) - 1:
                x = self.layer_act(x2)
            else:
                x = x2
            xs.append(x)
        if self.residual:
            x = torch.cat(xs, dim=1)
        x = self.activate(x)
        return x


def onehot_cols(n_row, row_indices, use_cuda=False) -> torch.Tensor:
    matrix = torch.zeros(size=[n_row, len(row_indices)])
    for i, ri in enumerate(row_indices):
        matrix[ri, i] = 1
    if use_cuda:
        matrix = matrix.cuda()
    return matrix


class GraphIsomorphismNetwork(nn.Module):
    def __init__(self, atom_dim: int, bond_dim: int, h_dim: int, n_layer=4, use_cuda=False):
        super(GraphIsomorphismNetwork, self).__init__()
        self.use_cuda = use_cuda
        self.v_linear = MLP(atom_dim, h_dim, hidden_dims=[h_dim], activation='tanh', use_cuda=use_cuda, bias=True)
        self.e_linear = MLP(bond_dim + 1, h_dim, hidden_dims=[h_dim], activation='tanh', use_cuda=use_cuda, bias=True)
        self.mlps = nn.ModuleList(
            [MLP(h_dim, h_dim, activation='leaky_relu', use_cuda=use_cuda) for _ in range(n_layer)]
        )

    def forward(self, atom_ftr: torch.Tensor, bond_ftr: torch.Tensor, vew1: torch.Tensor, vew2: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        hv_ftr = self.v_linear(atom_ftr)
        he_ftr = self.e_linear(bond_ftr)
        for mlp in self.mlps:
            he_hv_2_ftr = F.relu(he_ftr + (vew2.t() @ hv_ftr))
            hv_ftr = hv_ftr + (vew1 @ he_hv_2_ftr)
            hv_ftr = mlp(hv_ftr)

        he_ftr = torch.cat([vew1.t() @ hv_ftr, vew2.t() @ hv_ftr, he_ftr], dim=1)
        return hv_ftr, he_ftr

    @staticmethod
    def extend_graph(bond_ftr: torch.Tensor, pos: torch.Tensor, vew1: torch.Tensor, vew2: torch.Tensor, use_cuda=False
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_v, n_e = vew1.shape
        adj = vew1 @ vew2.t()
        adj3 = adj @ adj @ adj
        adj_ex = torch.logical_and(adj3 > 0, adj < 1e-3)
        indices = torch.nonzero(torch.flatten(adj_ex)).squeeze(1)
        n_ex = indices.shape[0]
        bond_ftr_ex = torch.zeros(size=[n_ex, bond_ftr.shape[1]])
        if use_cuda:
            bond_ftr_ex = bond_ftr_ex.cuda()
        vew1_ex = onehot_cols(n_v, [int(index / n_v) for index in indices], use_cuda=use_cuda)
        vew2_ex = onehot_cols(n_v, [int(index % n_v) for index in indices], use_cuda=use_cuda)
        bond_ftr = torch.cat([bond_ftr, bond_ftr_ex], dim=0)
        vew1 = torch.cat([vew1, vew1_ex], dim=1)
        vew2 = torch.cat([vew2, vew2_ex], dim=1)
        pos_1 = vew1.t() @ pos
        pos_2 = vew2.t() @ pos
        dis = torch.norm(pos_1 - pos_2, dim=1, keepdim=True)
        bond_ftr = torch.cat([bond_ftr, dis], dim=1)
        return bond_ftr, vew1, vew2

    @staticmethod
    def extend_graph_no_pos(bond_ftr: torch.Tensor, vew1: torch.Tensor, vew2: torch.Tensor,
                            use_cuda=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_v, n_e = vew1.shape
        adj = vew1 @ vew2.t()
        adj3 = adj @ adj @ adj
        adj_ex = torch.logical_and(adj3 > 0, adj < 1e-3)
        indices = torch.nonzero(torch.flatten(adj_ex)).squeeze(1)
        n_ex = indices.shape[0]
        bond_ftr_ex = torch.zeros(size=[n_ex, bond_ftr.shape[1]])
        if use_cuda:
            bond_ftr_ex = bond_ftr_ex.cuda()
        vew1_ex = onehot_cols(n_v, [int(index / n_v) for index in indices], use_cuda=use_cuda)
        vew2_ex = onehot_cols(n_v, [int(index % n_v) for index in indices], use_cuda=use_cuda)
        bond_ftr = torch.cat([bond_ftr, bond_ftr_ex], dim=0)
        vew1 = torch.cat([vew1, vew1_ex], dim=1)
        vew2 = torch.cat([vew2, vew2_ex], dim=1)
        dis = torch.norm(torch.zeros_like(bond_ftr), dim=1, keepdim=True)
        bond_ftr = torch.cat([bond_ftr, dis], dim=1)
        return bond_ftr, vew1, vew2

    @staticmethod
    def extend_graph_no_dis(vew1: torch.Tensor, vew2: torch.Tensor, use_cuda=False
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_v, n_e = vew1.shape
        adj = vew1 @ vew2.t()
        adj3 = adj @ adj @ adj
        adj_ex = torch.logical_and(adj3 > 0, adj < 1e-3)
        indices = torch.nonzero(torch.flatten(adj_ex)).squeeze(1)
        vew1_ex = onehot_cols(n_v, [int(index / n_v) for index in indices], use_cuda=use_cuda)
        vew2_ex = onehot_cols(n_v, [int(index % n_v) for index in indices], use_cuda=use_cuda)
        vew1 = torch.cat([vew1, vew1_ex], dim=1)
        vew2 = torch.cat([vew2, vew2_ex], dim=1)
        return vew1, vew2

    @staticmethod
    def edge_distances(pos: torch.Tensor, vew1: torch.Tensor, vew2: torch.Tensor, keepdim=True) -> torch.Tensor:
        pos_1 = vew1.t() @ pos
        pos_2 = vew2.t() @ pos
        dis = torch.norm(pos_1 - pos_2, dim=1, keepdim=keepdim)
        return dis
