import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple

from data.structures import MaskMatrices
from .option import ModelType, ConfType
from .AttentiveFP import AttentiveLayers
from .CVGAE import CVGAE
from .HamNet import HamNet
from .PhysChem import PhysChem


class MoleculeModel(nn.Module):
    def __init__(self, model_type: ModelType, atom_dim: int, bond_dim: int, config: Dict[str, Any], use_cuda=False):
        super(MoleculeModel, self).__init__()
        self.model_type = model_type
        if model_type == ModelType.HAM_NET:
            self.model = HamNet(atom_dim, bond_dim, config, use_cuda=use_cuda)
        elif model_type == ModelType.PHYS_CHEM:
            self.model = PhysChem(atom_dim, bond_dim, config, use_cuda=use_cuda)
        else:
            assert False, f'Undefined model_type {model_type} in net.models.MoleculeModel.'

    def forward(self, atom_ftr: torch.Tensor, bond_ftr: torch.Tensor, massive: torch.Tensor,
                mask_matrices: MaskMatrices, given_q_ftr: torch.Tensor = None,
                return_list: List[str] = None, extra_dict: Dict[str, Any] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        return_dict = {}
        if self.model_type == ModelType.HAM_NET:
            fingerprint, atom_positions, *_ = self.model.forward(
                atom_ftr=atom_ftr,
                bond_ftr=bond_ftr,
                massive=massive,
                mask_matrices=mask_matrices,
                given_q_ftr=given_q_ftr
            )
        elif self.model_type == ModelType.PHYS_CHEM:
            fingerprint, atom_positions, *_ = self.model.forward(
                atom_ftr=atom_ftr,
                bond_ftr=bond_ftr,
                massive=massive,
                mask_matrices=mask_matrices,
                given_q_ftr=given_q_ftr
            )
        else:
            assert False

        return fingerprint, atom_positions, return_dict
