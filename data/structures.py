import numpy as np
import torch
from typing import List, Tuple

from data.encode import encode_mols, get_massive_from_atom_features


class MaskMatrices:
    def __init__(self, mol_vertex_w: torch.Tensor, mol_vertex_b: torch.Tensor,
                 vertex_edge_w1: torch.Tensor, vertex_edge_w2: torch.Tensor,
                 vertex_edge_b1: torch.Tensor, vertex_edge_b2: torch.Tensor):
        self.mol_vertex_w = mol_vertex_w
        self.mol_vertex_b = mol_vertex_b
        self.vertex_edge_w1 = vertex_edge_w1
        self.vertex_edge_w2 = vertex_edge_w2
        self.vertex_edge_b1 = vertex_edge_b1
        self.vertex_edge_b2 = vertex_edge_b2

    def cuda_copy(self):
        return MaskMatrices(
            mol_vertex_w=self.mol_vertex_w.cuda(),
            mol_vertex_b=self.mol_vertex_b.cuda(),
            vertex_edge_w1=self.vertex_edge_w1.cuda(),
            vertex_edge_w2=self.vertex_edge_w2.cuda(),
            vertex_edge_b1=self.vertex_edge_b1.cuda(),
            vertex_edge_b2=self.vertex_edge_b2.cuda()
        )


class PackedMolGraph:
    def __init__(self, molecules: List):
        molecules_info, mask = encode_mols(molecules)
        # assert len(mask) == len(molecules)

        atom_ftr = np.vstack([molecules_info[m]['af'] for m in mask])
        bond_ftr = np.vstack([molecules_info[m]['bf'] for m in mask])
        massive = get_massive_from_atom_features(atom_ftr)
        n_mol = len(mask)
        ns_atom = [molecules_info[m]['af'].shape[0] for m in mask]
        ns_bond = [molecules_info[m]['bf'].shape[0] for m in mask]
        ms = []
        us = []
        vs = []
        for i, m in enumerate(mask):
            ms.extend([i] * ns_atom[i])
            prev_bonds = sum(ns_atom[:i])
            us.extend(molecules_info[m]['us'] + prev_bonds)
            vs.extend(molecules_info[m]['vs'] + prev_bonds)
        mol_vertex_w, mol_vertex_b = self.produce_mask_matrix(len(mask), ms)
        vertex_edge_w1, vertex_edge_b1 = self.produce_mask_matrix(sum(ns_atom), us)
        vertex_edge_w2, vertex_edge_b2 = self.produce_mask_matrix(sum(ns_atom), vs)

        self.mask = mask
        self.n_mol = n_mol
        self.n_bond = sum(ns_bond)
        self.n_atom = sum(ns_atom)
        self.atom_ftr = torch.FloatTensor(atom_ftr)
        self.bond_ftr = torch.FloatTensor(bond_ftr)
        self.massive = torch.FloatTensor(massive)
        mol_vertex_w = torch.FloatTensor(mol_vertex_w)
        mol_vertex_b = torch.FloatTensor(mol_vertex_b)
        vertex_edge_w1 = torch.FloatTensor(vertex_edge_w1)
        vertex_edge_b1 = torch.FloatTensor(vertex_edge_b1)
        vertex_edge_w2 = torch.FloatTensor(vertex_edge_w2)
        vertex_edge_b2 = torch.FloatTensor(vertex_edge_b2)
        self.mask_matrices = MaskMatrices(mol_vertex_w, mol_vertex_b,
                                          vertex_edge_w1, vertex_edge_w2,
                                          vertex_edge_b1, vertex_edge_b2)

    @staticmethod
    def produce_mask_matrix(n: int, s: list) -> Tuple[np.ndarray, np.ndarray]:
        s = np.array(s)
        mat = np.full([n, s.shape[0]], 0., dtype=np.int)
        mask = np.full([n, s.shape[0]], -1e6, dtype=np.int)
        for i in range(n):
            node_edge = s == i
            mat[i, node_edge] = 1
            mask[i, node_edge] = 0
        return mat, mask
