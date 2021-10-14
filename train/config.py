from typing import Dict, Any

from data.option import SupportedDataset
from net.option import ModelType
from net.HamNet.model_config import get_hamnet_default_config
from net.PhysChem.model_config import get_physchem_default_config

MODEL_CONFIG_FETCH_DICT = {
    ModelType.HAM_NET: get_hamnet_default_config,
    ModelType.PHYS_CHEM: get_physchem_default_config,
}


def get_default_config(dataset_name: SupportedDataset, model_type: ModelType) -> Dict[str, Any]:
    if model_type in MODEL_CONFIG_FETCH_DICT.keys():
        config = MODEL_CONFIG_FETCH_DICT[model_type](dataset_name)
    else:
        assert False, f'Unsupported model_type {model_type}.'

    return config
