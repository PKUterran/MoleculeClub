from data.option import SupportedDataset
from net.option import ConfType


DEFAULT_CONFIG = {
    # model
    'CLASSIFIER_HIDDENS': [],

    'INIT_GCN_H_DIMS': [128],
    'INIT_GCN_O_DIM': 128,
    'INIT_LSTM_LAYERS': 2,
    'INIT_LSTM_O_DIM': 128,

    'HV_DIM': 128,
    'HE_DIM': 64,
    'HM_DIM': 256,
    'MV_DIM': 128,
    'ME_DIM': 64,
    'MM_DIM': 256,
    'PQ_DIM': 3,
    'N_LAYER': 2,
    'N_HOP': 1,
    'N_ITERATION': 4,
    'N_GLOBAL': 2,
    'MESSAGE_TYPE': 'triplet',
    'UNION_TYPE': 'gru',
    'GLOBAL_TYPE': 'inductive',
    'DERIVATION_TYPE': 'newton',
    'TAU': 0.25,
    'DROPOUT': 0.5,

    'EPOCH': 300,
    'PACK': 20,
    'BATCH': 1,
    'LAMBDA': 100,
    'LR': 2e-6,
    'GAMMA': 0.995,
    'DECAY': 1e-5,

    'conf_type': ConfType.RDKIT
}

LIPOP_CONFIG = DEFAULT_CONFIG.copy()
LIPOP_CONFIG.update({
    'CLASSIFIER_HIDDENS': [],
    'HV_DIM': 256,
    'HE_DIM': 256,
    'HM_DIM': 256,
    'MV_DIM': 256,
    'ME_DIM': 256,
    'MM_DIM': 256,

    'N_LAYER': 2,
    'N_ITERATION': 1,
    'N_HOP': 1,
    'N_GLOBAL': 2,
    'DROPOUT': 0.2,

    'EPOCH': 400,
    'PACK': 2,
    'BATCH': 8,
    'LAMBDA': 0.1,
    'LR': 1e-4,
    'GAMMA': 0.995,
    'DECAY': 1e-5,
})

ESOL_CONFIG = DEFAULT_CONFIG.copy()
ESOL_CONFIG.update({
    'CLASSIFIER_HIDDENS': [],
    'HV_DIM': 160,
    'HE_DIM': 160,
    'HM_DIM': 160,
    'MV_DIM': 160,
    'ME_DIM': 160,
    'MM_DIM': 160,

    'N_LAYER': 2,
    'N_ITERATION': 4,
    'N_HOP': 1,
    'N_GLOBAL': 2,
    'DROPOUT': 0.2,

    'EPOCH': 800,
    'PACK': 8,
    'BATCH': 16,
    'LAMBDA': 0.1,
    'LR': 3e-3,
    'GAMMA': 0.995,
    'DECAY': 1e-5,
})

FREESOLV_CONFIG = DEFAULT_CONFIG.copy()
FREESOLV_CONFIG.update({
    'CLASSIFIER_HIDDENS': [],
    'HV_DIM': 120,
    'HE_DIM': 120,
    'HM_DIM': 120,
    'MV_DIM': 120,
    'ME_DIM': 120,
    'MM_DIM': 120,

    'N_LAYER': 2,
    'N_ITERATION': 4,
    'N_HOP': 1,
    'N_GLOBAL': 2,
    'DROPOUT': 0.2,

    'EPOCH': 400,
    'PACK': 1,
    'BATCH': 128,
    'LAMBDA': 0.1,
    'LR': 3e-3,
    'GAMMA': 0.995,
    'DECAY': 1e-5,
})

TOX21_CONFIG = DEFAULT_CONFIG.copy()
TOX21_CONFIG.update({
    'CLASSIFIER_HIDDENS': [],
    'HV_DIM': 200,
    'HE_DIM': 200,
    'HM_DIM': 200,
    'MV_DIM': 200,
    'ME_DIM': 200,
    'MM_DIM': 200,

    'N_LAYER': 2,
    'N_ITERATION': 4,
    'N_HOP': 1,
    'N_GLOBAL': 2,
    'DROPOUT': 0.5,

    'EPOCH': 300,
    'PACK': 1,
    'BATCH': 128,
    'LAMBDA': 0.1,
    'LR': 1e-4,
    'GAMMA': 0.99,
    'DECAY': 1e-5,
})
SARS_CONFIG = DEFAULT_CONFIG.copy()
SARS_CONFIG.update({
    'CLASSIFIER_HIDDENS': [],
    'HV_DIM': 200,
    'HE_DIM': 200,
    'HM_DIM': 200,
    'MV_DIM': 200,
    'ME_DIM': 200,
    'MM_DIM': 200,

    'N_LAYER': 2,
    'N_ITERATION': 4,
    'N_HOP': 1,
    'N_GLOBAL': 2,
    'DROPOUT': 0.5,

    'EPOCH': 300,
    'PACK': 1,
    'BATCH': 64,
    'LAMBDA': 0.1,
    'LR': 1e-4,
    'GAMMA': 0.99,
    'DECAY': 1e-5,
})

DATASET_CONFIG_DICT = {
    SupportedDataset.LIPOP: LIPOP_CONFIG,
    SupportedDataset.FREESOLV: FREESOLV_CONFIG,
    SupportedDataset.ESOL: ESOL_CONFIG,
    SupportedDataset.TOX21: TOX21_CONFIG,
    SupportedDataset.SARS: SARS_CONFIG,
}


def get_physchem_default_config(dataset_name: SupportedDataset) -> dict:
    if dataset_name in DATASET_CONFIG_DICT.keys():
        return DATASET_CONFIG_DICT[dataset_name]
    return DEFAULT_CONFIG
