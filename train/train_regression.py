import json
import os
import numpy as np
import torch
import torch.optim as optim
from typing import List, Dict, Any
from itertools import chain
from functools import reduce

from data.encode import num_atom_features, num_bond_features
from data.structures import MaskMatrices, PackedMolGraph
from data.load_data import GeoMolDataset, load_data
from data.option import SupportedDataset
from data.utils import set_seed
from net.utils.components import MLP
from net.models import MoleculeModel
from net.option import ModelType
from train.config import get_default_config


def train_prop(dataset_name: SupportedDataset, split_seed: int, dataset_token: str = None,
               model_type: ModelType = ModelType.PHYS_CHEM, train_seed=0,
               special_config: Dict[str, Any] = None,
               max_num=-1, force_save=False,
               use_cuda=False, use_tqdm=False):
    set_seed(train_seed)
    config = get_default_config(dataset_name, model_type)
    config.update(special_config)
    print('##### Config #####')
    for k, v in config.items():
        print(f'\t{k}: {v}')
    train_set, validate_set, test_set, info_dict = load_data(
        dataset_name=dataset_name, split_seed=split_seed, dataset_token=dataset_token,
        n_mol_per_pack=config['PACK'], max_num=max_num, force_save=force_save,
        use_tqdm=use_tqdm
    )

    model = MoleculeModel(
        model_type=model_type,
        atom_dim=num_atom_features(),
        bond_dim=num_bond_features(),
        config=config,
        use_cuda=use_cuda
    )
    classifier = MLP(
        in_dim=config['HM_DIM'],
        out_dim=info_dict['target_dim'],
        hidden_dims=config['CLASSIFIER_HIDDENS'],
        use_cuda=use_cuda,
        bias=True
    )
    if use_cuda:
        model.cuda()
        classifier.cuda()

    # initialize optimization
    parameters = list(chain(model.parameters(), classifier.parameters()))
    optimizer = optim.Adam(params=parameters, lr=config['LR'], weight_decay=config['DECAY'])
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=config['GAMMA'])
    print('##### Parameters #####')

    param_size = 0
    for name, param in chain(model.named_parameters(), classifier.named_parameters()):
        print(f'\t\t{name}: {param.shape}')
        param_size += reduce(lambda x, y: x * y, param.shape)
    print(f'\tNumber of parameters: {param_size}')

    # train
    epoch = 0
    logs: List[Dict[str, Any]] = []
    conf_type = config['CONF_TYPE']
