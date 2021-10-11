import torch
import torch.nn as nn
from typing import Union, Tuple, List, Dict, Any

from data.structures import MaskMatrices
from net.utils.components import GCN, MLP
from net.utils.functions import normalize_adj_rc


class _NaiveMPNN(nn.Module):
    def __init__(self, hv_dim, he_dim, out_dim, use_cuda=False):
        super(_NaiveMPNN, self).__init__()
        self.edge_attend = MLP(hv_dim + he_dim + hv_dim, 1, hidden_dims=[hv_dim], activation='sigmoid')
        self.list_gcn = nn.ModuleList([GCN(hv_dim, hv_dim, use_cuda=use_cuda) for _ in range(4)])
        self.gru_cell = nn.GRUCell(hv_dim, hv_dim)
        self.remap = MLP(hv_dim + hv_dim, out_dim, hidden_dims=[hv_dim], use_cuda=use_cuda, dropout=0.2)

    def forward(self, hv_ftr: torch.Tensor, he_ftr: torch.Tensor, mask_matrices: MaskMatrices,
                sample_ftr: torch.Tensor = None) -> torch.Tensor:
        if sample_ftr is not None:
            hvs_ftr = hv_ftr + sample_ftr
        else:
            hvs_ftr = hv_ftr
        vew1 = mask_matrices.vertex_edge_w1  # shape [n_vertex, n_edge]
        vew2 = mask_matrices.vertex_edge_w2  # shape [n_vertex, n_edge]
        vew_u = torch.cat([vew1, vew2], dim=1)  # shape [n_vertex, 2 * n_edge]
        vew_v = torch.cat([vew2, vew1], dim=1)  # shape [n_vertex, 2 * n_edge]
        hv_u_ftr = vew_u.t() @ hv_ftr  # shape [2 * n_edge, hv_dim]
        hv_v_ftr = vew_v.t() @ hv_ftr  # shape [2 * n_edge, hv_dim]
        he2_ftr = torch.cat([he_ftr, he_ftr])  # shape [2 * n_edge, he_dim]
        uev_ftr = torch.cat([hv_u_ftr, he2_ftr, hv_v_ftr], dim=1)  # shape [2 * n_edge, hv_dim + he_dim + hv_dim]

        edge_weight = self.edge_attend(uev_ftr)  # shape [2 * n_edge, 1]
        adj = vew_u @ (vew_v * edge_weight.view(-1)).t()  # shape [n_vertex, n_vertex]
        adj = normalize_adj_rc(adj)
        hidden = hvs_ftr
        for gcn in self.list_gcn:
            message = gcn.forward(hidden, adj)  # shape [n_vertex, hv_dim]
            hidden = self.gru_cell.forward(message, hidden)
        out = self.remap(torch.cat([hv_ftr, hidden], dim=1))  # shape [n_vertex, out_dim]
        return out


class _CVGAECore(nn.Module):
    def __init__(self, hv_dim: int, he_dim: int, use_cuda=False):
        super(_CVGAECore, self).__init__()
        self.use_cuda = use_cuda
        self.hv_dim = hv_dim
        self.vp_act = nn.Tanh()
        self.vp_linear = nn.Linear(hv_dim + 3, hv_dim, bias=True)
        self.prior_mpnn = _NaiveMPNN(hv_dim, he_dim, 2 * hv_dim, use_cuda=use_cuda)
        self.post_mpnn = _NaiveMPNN(hv_dim, he_dim, 2 * hv_dim, use_cuda=use_cuda)
        self.pred_mpnn = _NaiveMPNN(hv_dim, he_dim, 2 * hv_dim, use_cuda=use_cuda)

    def forward(self, hv_ftr: torch.Tensor, he_ftr: torch.Tensor,
                mask_matrices: MaskMatrices, is_training=False, given_pos: torch.Tensor = None
                ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        assert not is_training or given_pos is not None
        prior_z_out = self.prior_mpnn.forward(hv_ftr, he_ftr, mask_matrices)
        prior_z_mu, prior_z_lsgms = torch.split(prior_z_out, [self.hv_dim, self.hv_dim], dim=1)
        prior_z_sample = self._draw_sample(prior_z_mu, prior_z_lsgms)

        if is_training or given_pos is not None:
            hvp_ftr = self.vp_act(self.vp_linear(torch.cat([hv_ftr, given_pos], dim=1)))
            post_z_out = self.post_mpnn.forward(hvp_ftr, he_ftr, mask_matrices)
            post_z_mu, post_z_lsgms = torch.split(post_z_out, [self.hv_dim, self.hv_dim], dim=1)
            post_z_sample = self._draw_sample(post_z_mu, post_z_lsgms)
            post_x_out = self.pred_mpnn.forward(hv_ftr, he_ftr, mask_matrices, sample_ftr=post_z_sample)
            if not is_training:  # evaluating with UFF
                return post_x_out
            # training with ground truth
            klds_z = self._kld(post_z_mu, post_z_lsgms, prior_z_mu, prior_z_lsgms)
            klds_0 = self._kld_zero(prior_z_mu, prior_z_lsgms)
            return post_x_out, klds_z, klds_0
        else:  # evaluating without UFF
            prior_x_out = self.pred_mpnn.forward(hv_ftr, he_ftr, mask_matrices, sample_ftr=prior_z_sample)
            return prior_x_out

    def _draw_sample(self, mu: torch.Tensor, lsgms: torch.Tensor, T=1):
        epsilon = torch.normal(torch.zeros(size=lsgms.shape, dtype=torch.float32), 1.)
        if self.use_cuda:
            epsilon = epsilon.cuda()
        sample = torch.mul(torch.exp(0.5 * lsgms) * T, epsilon)
        sample = torch.add(mu, sample)
        return sample

    @staticmethod
    def _kld(mu0, lsgm0, mu1, lsgm1):
        var0 = torch.exp(lsgm0)
        var1 = torch.exp(lsgm1)
        a = torch.div(var0 + 1e-5, var1 + 1e-5)
        b = torch.div(torch.pow(mu1 - mu0, 2), var1 + 1e-5)
        c = torch.log(torch.div(var1 + 1e-5, var0 + 1e-5) + 1e-5)
        kld = 0.5 * torch.sum(a + b - 1 + c, dim=1)
        return kld

    @staticmethod
    def _kld_zero(mu, lsgm):
        a = torch.exp(lsgm) + torch.pow(mu, 2)
        b = 1 + lsgm
        kld = 0.5 * torch.sum(a - b, dim=1)
        return kld


class CVGAE(nn.Module):
    def __init__(self, hv_dim: int, he_dim: int, pos_dim: int, need_momentum=False, use_cuda=False, p_dropout=0.0):
        super(CVGAE, self).__init__()
        self.hv_dim = hv_dim
        self.he_dim = he_dim
        self.pos_dim = pos_dim
        self.need_momentum = need_momentum
        self.use_cuda = use_cuda
        self.dropout = nn.Dropout(p_dropout)
        self.cvgae = _CVGAECore(
            hv_dim=self.hv_dim,
            he_dim=self.he_dim,
            use_cuda=self.use_cuda
        )
        self.mlp = MLP(
            in_dim=2 * self.hv_dim,
            out_dim=self.pos_dim * 2 if self.need_momentum else self.pos_dim,
            use_cuda=self.use_cuda,
            bias=False
        )

    def forward(self, hv_ftr: torch.Tensor, he_ftr: torch.Tensor, mask_matrices: MaskMatrices,
                is_training=False, target_conf: torch.Tensor = None, rdkit_conf: torch.Tensor = None
                ) -> Tuple[Union[torch.Tensor, None], torch.Tensor, Dict[str, Any]]:
        return_dict = {}
        if is_training:
            post_x_out, klds_z, klds_0 = self.cvgae.forward(hv_ftr, he_ftr, mask_matrices,
                                                            is_training, given_pos=target_conf)
            pq_ftr = self.mlp(post_x_out)
            return_dict['kld_z_loss'] = torch.mean(klds_z)
            return_dict['kld_0_loss'] = torch.mean(klds_0)
        else:
            post_x_out = self.cvgae.forward(hv_ftr, he_ftr, mask_matrices,
                                            is_training, given_pos=rdkit_conf)
            pq_ftr = self.mlp(post_x_out)

        if self.need_momentum:
            p_ftr, q_ftr = pq_ftr[:, :self.pos_dim], pq_ftr[:, self.pos_dim:]
            return p_ftr, q_ftr, return_dict
        else:
            return None, pq_ftr, return_dict
