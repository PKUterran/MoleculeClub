from enum import Enum


# path
class ModelType(Enum):
    ATTENTIVE_FP = 'AttentiveFP'
    CVGAE = 'CVGAE'
    HAM_NET = 'HamNet'
    PHYS_CHEM = 'PhysChem'


class ConfType(Enum):
    NONE = 0,
    RDKIT = 1,
    NEWTON = 2,
    ONLY = 3,
    NEWTON_RGT = 4,
    REAL = 5,
    SINGLE_CHANNEL = 6,
