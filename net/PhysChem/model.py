from .layers import *
from enum import Enum


class ConfType(Enum):
    NONE = 0,
    RDKIT = 1,
    NEWTON = 2,
    ONLY = 3,
    NEWTON_RGT = 4,
    REAL = 5,
    SINGLE_CHANNEL = 6,


class PhysChem(nn.Module):
    def __init__(self, atom_dim: int, bond_dim: int, config: dict,
                 use_cuda=False):
        super(PhysChem, self).__init__()
        hv_dim = config['HV_DIM']
        he_dim = config['HE_DIM']
        hm_dim = config['HM_DIM']
        mv_dim = config['MV_DIM']
        me_dim = config['ME_DIM']
        mm_dim = config['MM_DIM']
        pq_dim = config['PQ_DIM']
        self.n_layer = config['N_LAYER']
        self.n_iteration = config['N_ITERATION']
        n_hop = config['N_HOP']
        n_global = config['N_GLOBAL']
        message_type = config['MESSAGE_TYPE']
        union_type = config['UNION_TYPE']
        global_type = config['GLOBAL_TYPE']
        tau = config['TAU']
        p_dropout = config['DROPOUT']
        self.use_cuda = use_cuda

        self.conf_type = config['CONF_TYPE']
        self.need_derive = self.conf_type not in [ConfType.NONE, ConfType.RDKIT, ConfType.REAL, ConfType.SINGLE_CHANNEL]
        self.need_mp = self.conf_type is not ConfType.ONLY

        self.initializer = Initializer(
            atom_dim=atom_dim,
            bond_dim=bond_dim,
            hv_dim=hv_dim,
            he_dim=he_dim,
            p_dim=pq_dim,
            q_dim=pq_dim,
            gcn_h_dims=config['INIT_GCN_H_DIMS'],
            gcn_o_dim=config['INIT_GCN_O_DIM'],
            lstm_layers=config['INIT_LSTM_LAYERS'],
            lstm_o_dim=config['INIT_LSTM_O_DIM'],
            use_cuda=use_cuda
        )
        if self.need_mp:
            self.chem_net = ChemNet(
                hv_dim=hv_dim,
                he_dim=he_dim,
                mv_dim=mv_dim,
                me_dim=me_dim,
                p_dim=pq_dim,
                q_dim=pq_dim,
                hops=n_hop,
                use_cuda=use_cuda,
                p_dropout=p_dropout,
                message_type=message_type,
                union_type=union_type
            )
        if self.need_derive:
            self.phys_net = PhysNet(
                hv_dim=hv_dim,
                he_dim=he_dim,
                pq_dim=pq_dim,
                tau=tau,
                use_cuda=use_cuda,
                p_dropout=p_dropout
            )
        if global_type == 'recurrent':
            self.fingerprint_gen = RecFingerprintGenerator(
                hm_dim=hm_dim,
                hv_dim=hv_dim,
                mm_dim=mm_dim,
                iteration=n_global,
                use_cuda=use_cuda,
                p_dropout=p_dropout
            )
        else:
            self.fingerprint_gen = FingerprintGenerator(
                hm_dim=hm_dim,
                hv_dim=hv_dim,
                mm_dim=mm_dim,
                iteration=n_global,
                use_cuda=use_cuda,
                p_dropout=p_dropout
            )
        if self.conf_type == ConfType.SINGLE_CHANNEL:
            self.conformation_encode = MLP(hv_dim, pq_dim, hidden_dims=[hv_dim], use_cuda=use_cuda)
        if pq_dim != 3:
            self.conformation_gen = MLP(pq_dim, 3, use_cuda=use_cuda)
        else:
            self.conformation_gen = lambda x: x

    def forward(self, atom_ftr: torch.Tensor, bond_ftr: torch.Tensor, massive: torch.Tensor,
                mask_matrices: MaskMatrices,
                given_q_ftr: torch.Tensor = None,
                return_local_alignment=False, return_global_alignment=False, return_derive=False
                ) -> Tuple[torch.Tensor, List[torch.Tensor],
                           List[List[np.ndarray]], List[np.ndarray], List[np.ndarray],
                           List[np.ndarray], List[np.ndarray]]:
        hv_ftr, he_ftr, p_ftr, q_ftr = self.initializer.forward(atom_ftr, bond_ftr, mask_matrices, not self.need_derive)

        if self.conf_type in [ConfType.NONE, ConfType.SINGLE_CHANNEL]:
            p_ftr = q_ftr = torch.zeros(size=[atom_ftr.shape[0], 3], dtype=torch.float32)
            if self.use_cuda:
                p_ftr, q_ftr = p_ftr.cuda(), q_ftr.cuda()
        elif self.conf_type in [ConfType.RDKIT, ConfType.REAL]:
            p_ftr, q_ftr = torch.zeros(size=[atom_ftr.shape[0], 3], dtype=torch.float32), given_q_ftr
            if self.use_cuda:
                p_ftr = p_ftr.cuda()

        conformations = [self.conformation_gen(q_ftr)]
        list_alignments = []
        list_he_ftr = []
        list_p_ftr = []
        list_q_ftr = []
        if return_derive:
            list_p_ftr.append(self.decentralized_p_ftr(p_ftr, massive, mask_matrices).cpu().detach().numpy())
            list_q_ftr.append(q_ftr.cpu().detach().numpy())
        for i in range(self.n_layer):
            t_p_ftr, t_q_ftr = p_ftr, q_ftr
            if self.need_derive:
                for j in range(self.n_iteration):
                    p_ftr, q_ftr = self.phys_net.forward(hv_ftr, he_ftr, massive, p_ftr, q_ftr, mask_matrices)
                    conformations.append(self.conformation_gen(q_ftr))
                    if return_derive:
                        list_p_ftr.append(
                            self.decentralized_p_ftr(p_ftr, massive, mask_matrices).cpu().detach().numpy())
                        list_q_ftr.append(q_ftr.cpu().detach().numpy())

            if self.need_mp:
                hv_ftr, he_ftr, alignments = self.chem_net.forward(hv_ftr, he_ftr, t_p_ftr, t_q_ftr,
                                                                   mask_matrices, return_local_alignment)
                list_alignments.append(alignments)
            list_he_ftr.append(he_ftr.cpu().detach().numpy())

        fingerprint, global_alignments = self.fingerprint_gen.forward(hv_ftr, mask_matrices, return_global_alignment)
        if self.conf_type == ConfType.SINGLE_CHANNEL:
            q_ftr = self.conformation_encode(hv_ftr)
        conformations.append(self.conformation_gen(q_ftr))
        return fingerprint, conformations, list_alignments, global_alignments, list_he_ftr, list_p_ftr, list_q_ftr

    @staticmethod
    def decentralized_p_ftr(p_ftr: torch.Tensor, massive: torch.Tensor, mask_matrices: MaskMatrices) -> torch.Tensor:
        mvw = mask_matrices.mol_vertex_w
        mp = mvw @ p_ftr
        mm = mvw @ massive
        mv = mp / mm
        return p_ftr - (mvw.t() @ mv) * massive
