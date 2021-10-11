import torch
import torch.nn as nn

from .layers import LstmPQEncoder
from .hamiltion import DissipativeHamiltonianDerivation
from data.structures import MaskMatrices


class HamiltonEngine(nn.Module):
    def __init__(self, atom_dim, bond_dim, config, use_cuda=True):
        super(HamiltonEngine, self).__init__()
        self.hv_dim = 64
        self.he_dim = 16
        self.p_dim = config['PQ_DIM']
        self.q_dim = config['PQ_DIM']
        self.layers = config['HGN_LAYERS']
        self.tau = config['TAU']
        self.dropout = config['DROPOUT']
        self.dissipate = config['DISSIPATE']
        self.use_lstm = config['LSTM']
        self.use_cuda = use_cuda
        self.v_linear = nn.Linear(atom_dim, self.hv_dim, bias=True)
        self.v_act = nn.Tanh()
        self.e_linear = nn.Linear(bond_dim, self.he_dim, bias=True)
        self.e_act = nn.Tanh()
        self.e_encoder = nn.Linear(self.hv_dim + self.he_dim + self.hv_dim, 1)
        self.pq_encoder = LstmPQEncoder(self.hv_dim, self.p_dim,
                                        use_cuda=use_cuda, use_lstm=self.use_lstm)
        self.derivation = DissipativeHamiltonianDerivation(self.hv_dim, self.he_dim, self.p_dim, self.q_dim,
                                                           use_cuda=use_cuda, dropout=self.dropout)

    def forward(self, atom_ftr: torch.Tensor, bond_ftr: torch.Tensor, massive: torch.Tensor,
                mask_matrices: MaskMatrices, return_multi=False):
        hv_ftr = self.v_act(self.v_linear(atom_ftr))
        he_ftr = self.e_act(self.e_linear(bond_ftr))
        mvw = mask_matrices.mol_vertex_w
        vew1 = mask_matrices.vertex_edge_w1
        vew2 = mask_matrices.vertex_edge_w2
        u_e_v_features = torch.cat([vew1.t() @ hv_ftr, he_ftr, vew2.t() @ hv_ftr], dim=1)
        e_weight = torch.diag(torch.sigmoid(self.e_encoder(u_e_v_features)).view([-1]))
        e = vew1 @ e_weight @ vew2.t()
        p0, q0 = self.pq_encoder(hv_ftr, mvw, e)
        ps = [p0]
        qs = [q0]
        s_losses = []
        c_losses = []
        h = None
        d = None

        for i in range(self.layers):
            dp, dq, h, d = self.derivation.forward(hv_ftr, he_ftr, massive, ps[i], qs[i], mask_matrices,
                                                   return_energy=True, dissipate=self.dissipate)
            ps.append(ps[i] + self.tau * dp)
            qs.append(qs[i] + self.tau * dq)

            s_losses.append((dq - ps[i]).norm())
            c_losses.append((mvw @ (ps[i + 1] - ps[i])).norm())

        s_loss = sum(s_losses)
        c_loss = sum(c_losses)
        if self.dissipate:
            final_p = ps[-1]
            final_q = qs[-1]
        else:
            final_p = sum(ps) / len(ps)
            final_q = sum(qs) / len(qs)

        if return_multi:
            return ps, qs, s_loss, c_loss, h, d
        return final_p, final_q, s_loss, c_loss, h, d
