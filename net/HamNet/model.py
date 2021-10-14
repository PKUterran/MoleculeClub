import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

from data.structures import MaskMatrices
from net.PhysChem.layers import Initializer, ChemNet, FingerprintGenerator
from .pretrain import HamiltonEngine


class HamNet(nn.Module):
    def __init__(self, atom_dim: int, bond_dim: int, config: Dict[str, Any], use_cuda=False):
        super(HamNet, self).__init__()
        self.ham_eng_ = [HamiltonEngine(atom_dim, bond_dim, config, use_cuda=use_cuda)]
        self.ham_eng_[0].load_state_dict(torch.load(config['STATE_DICT_PATH']))

        self.hv_dim = config['HV_DIM']
        self.mv_dim = config['MV_DIM']
        self.he_dim = config['HE_DIM']
        self.me_dim = config['ME_DIM']
        self.hm_dim = config['HM_DIM']
        self.mm_dim = config['MM_DIM']
        self.p_dim = self.q_dim = config['PQ_DIM']
        self.hops = config['N_HOP']
        self.iteration = config['N_GLOBAL']
        self.p_dropout = config['DROPOUT']

        self.initializer = Initializer(
            atom_dim=atom_dim, bond_dim=bond_dim,
            hv_dim=self.hv_dim, he_dim=self.he_dim,
            p_dim=self.p_dim, q_dim=self.q_dim,
            gcn_h_dims=config['INIT_GCN_H_DIMS'],
            gcn_o_dim=config['INIT_GCN_O_DIM'],
            lstm_layers=config['INIT_LSTM_LAYERS'],
            lstm_o_dim=config['INIT_LSTM_O_DIM'],
            use_cuda=use_cuda
        )
        self.mpnn = ChemNet(
            hv_dim=self.hv_dim,
            he_dim=self.he_dim,
            mv_dim=self.mv_dim,
            me_dim=self.me_dim,
            p_dim=self.p_dim,
            q_dim=self.q_dim,
            hops=self.hops,
            use_cuda=use_cuda,
            p_dropout=self.p_dropout,
            message_type='naive',
            union_type='gru'
        )
        self.readout = FingerprintGenerator(
            hm_dim=self.hm_dim,
            hv_dim=self.hv_dim,
            mm_dim=self.mm_dim,
            iteration=self.iteration,
            use_cuda=use_cuda,
            p_dropout=self.p_dropout
        )

    def forward(self, atom_ftr: torch.Tensor, bond_ftr: torch.Tensor, massive: torch.Tensor,
                mask_matrices: MaskMatrices, given_q_ftr: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if given_q_ftr:
            q_ftr = given_q_ftr
            p_ftr = torch.zeros_like(q_ftr)
        else:
            p_ftr, q_ftr, *_ = self.ham_eng_[0].forward(
                atom_ftr, bond_ftr, massive, mask_matrices, return_multi=False)

        hv_ftr, he_ftr, *_ = self.initializer.forward(atom_ftr, bond_ftr, mask_matrices, pq_none=True)
        hv_ftr, he_ftr, _ = self.mpnn.forward(hv_ftr, he_ftr, p_ftr, q_ftr, mask_matrices)
        hm_ftr, _ = self.readout.forward(hv_ftr, mask_matrices)
        return hm_ftr, q_ftr
