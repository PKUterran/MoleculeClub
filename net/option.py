from enum import Enum


class ModelType(Enum):
    ATTENTIVE_FP = 'attentive_fp'
    CVGAE = 'cvgae'
    HAM_NET = 'ham_net'
    PHYS_CHEM = 'phys_chem'


class ConfType(Enum):
    NONE = 0,
    RDKIT = 1,
    NEWTON = 2,
    ONLY = 3,
    NEWTON_RGT = 4,
    REAL = 5,
    SINGLE_CHANNEL = 6,
