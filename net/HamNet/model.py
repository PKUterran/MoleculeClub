import torch
import torch.nn as nn
from typing import Dict, Any

from net.PhysChem.layers import Initializer, ChemNet, FingerprintGenerator
from .pretrain import HamiltonEngine

STATE_DICT_PATH = 'net/HamNet/state/hameng.pt'


class HamNet(nn.Module):
    def __init__(self, atom_dim: int, bond_dim: int,
                 config: Dict[str, Any], use_cuda=False):
        super(HamNet, self).__init__()
        self.ham_eng_ = [HamiltonEngine(atom_dim, bond_dim, config, use_cuda=use_cuda)]
        self.ham_eng_[0].load_state_dict(torch.load(STATE_DICT_PATH))

        self.hv_dim = config['HV_DIM']
        self.mv_dim = config['MV_DIM']
        self.he_dim = config['HE_DIM']
        self.me_dim = config['ME_DIM']
        self.hm_dim = config['HM_DIM']
        self.mm_dim = config['MM_DIM']
        self.p_dim = self.q_dim = config['PQ_DIM']
        self.gcn_h_dims = config['GCN_H_DIMS']
        self.gcn_o_dim = config['GCN_O_DIM']
        self.lstm_layers = config['LSTM_LAYERS']
        self.lstm_o_dim = config['LSTM_O_DIM']
        self.hops = config['HOPS']
        self.iteration = config['ITERATION']
        self.p_dropout = config['DROPOUT']

        self.initializer = Initializer(
            atom_dim=atom_dim, bond_dim=bond_dim,
            hv_dim=self.hv_dim, he_dim=self.he_dim,
            p_dim=self.p_dim, q_dim=self.q_dim,
            gcn_h_dims=self.gcn_h_dims, gcn_o_dim=self.gcn_o_dim,
            lstm_layers=self.lstm_layers,
            lstm_o_dim=self.lstm_o_dim,
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

    def forward(self):
        pass
