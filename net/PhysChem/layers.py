from net.utils.components import *
from .components import *
from .newton import NewtonianDerivation
from net.utils import normalize_adj_r
from typing import Union, List


class Initializer(nn.Module):
    def __init__(self, atom_dim: int, bond_dim: int, hv_dim: int, he_dim: int, p_dim: int, q_dim: int,
                 gcn_h_dims: List[int], gcn_o_dim: int, lstm_layers: int, lstm_o_dim: int,
                 use_cuda=False):
        super(Initializer, self).__init__()
        self.use_cuda = use_cuda
        self.p_dim = p_dim

        self.v_linear = nn.Linear(atom_dim, hv_dim, bias=True)
        self.v_act = nn.Tanh()
        self.e_linear = nn.Linear(bond_dim, he_dim, bias=True)
        self.e_act = nn.Tanh()
        self.a_linear = nn.Linear(he_dim, 1, bias=True)
        self.a_act = nn.Sigmoid()
        self.gcn = GCN(hv_dim, gcn_o_dim, gcn_h_dims, use_cuda=use_cuda, residual=True)
        if lstm_o_dim == 0:
            self.remap = False
            self.lstm_encoder = LSTMEncoder(hv_dim + sum(gcn_h_dims) + gcn_o_dim, p_dim + q_dim, layers=lstm_layers)
        else:
            self.remap = True
            self.lstm_encoder = LSTMEncoder(hv_dim + sum(gcn_h_dims) + gcn_o_dim, lstm_o_dim, layers=lstm_layers)
            self.lstm_act = nn.Tanh()
            self.lstm_remap = nn.Linear(lstm_o_dim, p_dim + q_dim)

    def forward(self, atom_ftr: torch.Tensor, bond_ftr: torch.Tensor,
                mask_matrices: MaskMatrices, pq_none=False
                ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, None, None]
    ]:
        vew1 = mask_matrices.vertex_edge_w1
        vew2 = mask_matrices.vertex_edge_w2

        hv_ftr = self.v_act(self.v_linear(atom_ftr))
        he_ftr = self.e_act(self.e_linear(bond_ftr))
        if pq_none:
            return hv_ftr, he_ftr, None, None

        a = self.a_act(self.a_linear(he_ftr))
        adj_d = vew1 @ torch.diag(torch.reshape(a, [-1])) @ vew2.t()
        adj = adj_d + adj_d.t()
        norm_adj = normalize_adj_r(adj)
        hv_neighbor_ftr = self.gcn(hv_ftr, norm_adj)
        if self.remap:
            pq_ftr = self.lstm_remap(self.lstm_act(self.lstm_encoder(hv_neighbor_ftr, mask_matrices)))
        else:
            pq_ftr = self.lstm_encoder(hv_neighbor_ftr, mask_matrices)
        p_ftr, q_ftr = pq_ftr[:, :self.p_dim], pq_ftr[:, self.p_dim:]

        return hv_ftr, he_ftr, p_ftr, q_ftr


class ChemNet(nn.Module):
    def __init__(self, hv_dim: int, he_dim: int, mv_dim: int, me_dim: int, p_dim:int, q_dim: int, hops: int,
                 use_cuda=False, p_dropout=0.0,
                 message_type='naive', union_type='gru'):
        super(ChemNet, self).__init__()
        self.use_cuda = use_cuda
        self.message_type = message_type
        self.union_type = union_type
        self.hops = hops

        if message_type == 'naive':
            self.messages = nn.ModuleList([
                NaiveDynMessage(hv_dim, he_dim, mv_dim, me_dim, p_dim, q_dim, use_cuda, p_dropout)
                for _ in range(hops)
            ])
        elif message_type == 'triplet':
            self.messages = nn.ModuleList([
                TripletAttnDynMessage(hv_dim, he_dim, mv_dim, me_dim, p_dim, q_dim, use_cuda, p_dropout)
                for _ in range(hops)
            ])
        elif message_type == 'triplet-mean':
            self.messages = nn.ModuleList([
                TripletDynMessage(hv_dim, he_dim, mv_dim, me_dim, p_dim, q_dim, use_cuda, p_dropout)
                for _ in range(hops)
            ])
        else:
            assert False, 'Undefined message type {} in net.layers.ChemNet'.format(message_type)

        if union_type == 'naive':
            self.unions_v = nn.ModuleList([NaiveUnion(hv_dim, mv_dim, use_cuda) for _ in range(hops)])
            self.unions_e = nn.ModuleList([NaiveUnion(he_dim, me_dim, use_cuda) for _ in range(hops)])
        elif union_type == 'gru':
            self.unions_v = nn.ModuleList([GRUUnion(hv_dim, mv_dim, use_cuda) for _ in range(hops)])
            self.unions_e = nn.ModuleList([GRUUnion(he_dim, me_dim, use_cuda) for _ in range(hops)])
        else:
            assert False, 'Undefined union type {} in net.layers.ChemNet'.format(union_type)

    def forward(self, hv_ftr: torch.Tensor, he_ftr: torch.Tensor, p_ftr: torch.Tensor, q_ftr: torch.Tensor,
                mask_matrices: MaskMatrices,
                return_alignment=False) -> Tuple[torch.Tensor, torch.Tensor, List[np.ndarray]]:
        alignments = []
        for i in range(self.hops):
            mv_ftr, me_ftr, alignment = self.messages[i].forward(hv_ftr, he_ftr, p_ftr, q_ftr,
                                                                 mask_matrices, return_alignment)
            hv_ftr = self.unions_v[i](hv_ftr, mv_ftr)
            he_ftr = self.unions_e[i](he_ftr, me_ftr)
            alignments.append(alignment)
        return hv_ftr, he_ftr, alignments


class PhysNet(nn.Module):
    def __init__(self, hv_dim: int, he_dim: int, pq_dim: int, tau: float, use_cuda=False, p_dropout=0.0,
                 instant_mode=False):
        super(PhysNet, self).__init__()
        self.tau = tau
        self.use_cuda = use_cuda
        self.instant_mode = instant_mode

        self.derivation = NewtonianDerivation(
            v_dim=hv_dim,
            e_dim=he_dim,
            pq_dim=pq_dim,
            use_cuda=use_cuda,
            p_dropout=p_dropout
        )

    def forward(self, hv_ftr: torch.Tensor, he_ftr: torch.Tensor,
                massive: torch.Tensor, p_ftr: torch.Tensor, q_ftr: torch.Tensor, mask_matrices: MaskMatrices
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        dp, dq, _ = self.derivation.forward(
            v=hv_ftr, e=he_ftr, m=massive,
            p=p_ftr, q=q_ftr,
            mask_matrices=mask_matrices
        )
        if self.instant_mode:
            p_ftr_ = dp
            q_ftr_ = q_ftr + (dp / massive) * self.tau
        else:
            p_ftr_ = p_ftr + dp * self.tau
            q_ftr_ = q_ftr + dq * self.tau
        return p_ftr_, q_ftr_


class FingerprintGenerator(nn.Module):
    def __init__(self, hm_dim: int, hv_dim: int, mm_dim: int, iteration: int,
                 use_cuda=False, p_dropout=0.0):
        super(FingerprintGenerator, self).__init__()
        self.use_cuda = use_cuda

        self.vertex2mol = nn.Linear(hv_dim, hm_dim, bias=True)
        self.vm_act = nn.LeakyReLU()
        self.readouts = nn.ModuleList([
            GlobalReadout(hm_dim, hv_dim, mm_dim, use_cuda, p_dropout)
            for _ in range(iteration)])
        self.unions = nn.ModuleList([
            GRUUnion(hm_dim, mm_dim, use_cuda)
            for _ in range(iteration)])
        self.iteration = iteration

    def forward(self, hv_ftr: torch.Tensor,
                mask_matrices: MaskMatrices,
                return_alignment=False) -> Tuple[torch.Tensor, List[np.ndarray]]:
        # initialize molecule features with mean of vertex features
        mvw = mask_matrices.mol_vertex_w
        norm_mvw = mvw / torch.sum(mvw, dim=-1, keepdim=True)
        hm_ftr = norm_mvw @ self.vm_act(self.vertex2mol(hv_ftr))

        # iterate
        alignments = []
        for i in range(self.iteration):
            mm_ftr, alignment = self.readouts[i].forward(hm_ftr, hv_ftr, mask_matrices, return_alignment)
            hm_ftr = self.unions[i].forward(hm_ftr, mm_ftr)
            alignments.append(alignment)

        return hm_ftr, alignments


class RecFingerprintGenerator(nn.Module):
    def __init__(self, hm_dim: int, hv_dim: int, mm_dim: int, iteration: int,
                 use_cuda=False, p_dropout=0.0):
        super(RecFingerprintGenerator, self).__init__()
        self.use_cuda = use_cuda

        self.vertex2mol = nn.Linear(hv_dim, hm_dim, bias=True)
        self.vm_act = nn.LeakyReLU()
        self.readout = GlobalReadout(hm_dim, hv_dim, mm_dim, use_cuda, p_dropout)
        self.union = GRUUnion(hm_dim, mm_dim, use_cuda)
        self.iteration = iteration

    def forward(self, hv_ftr: torch.Tensor,
                mask_matrices: MaskMatrices,
                return_alignment=False) -> Tuple[torch.Tensor, List[np.ndarray]]:
        # initialize molecule features with mean of vertex features
        mvw = mask_matrices.mol_vertex_w
        norm_mvw = mvw / torch.sum(mvw, dim=-1, keepdim=True)
        hm_ftr = norm_mvw @ self.vm_act(self.vertex2mol(hv_ftr))

        # iterate
        alignments = []
        for i in range(self.iteration):
            mm_ftr, alignment = self.readout.forward(hm_ftr, hv_ftr, mask_matrices, return_alignment)
            hm_ftr = self.union.forward(hm_ftr, mm_ftr)
            alignments.append(alignment)

        return hm_ftr, alignments


class ConformationGenerator(nn.Module):
    def __init__(self, q_dim: int, h_dims: list,
                 p_dropout=0.0):
        super(ConformationGenerator, self).__init__()
        self.mlp = MLP(q_dim, 3, h_dims, dropout=p_dropout)

    def forward(self, q_ftr: torch.Tensor) -> torch.Tensor:
        conf3d = self.mlp(q_ftr)
        return conf3d
