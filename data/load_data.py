import os
import json
import numpy as np
import tqdm
import pickle
import torch
import rdkit.Chem as Chem

from typing import List, Tuple, Dict, Any
from torch.utils.data import Dataset, Subset
from rdkit.Chem import AllChem

from data.option import SupportedDataset
from data.structures import PackedMolGraph
from data.utils import make_dir, split_by_interval, target_positions, rdkit_positions, get_mean_std
from data.split import generate_split
from data.QM7.load_qm7 import load_qm7
from data.Lipop.load_lipop import load_lipop
from data.FreeSolv.load_freesolv import load_freesolv
from data.ESOL.load_esol import load_esol
from data.TOX21.load_tox21 import load_tox21
from data.sars.load_sars import load_sars


class GeoMolDataset(Dataset):
    def __init__(self,
                 list_packed_mol_graph: List[PackedMolGraph],
                 list_smiles_set: List[List[str]],
                 list_properties: List[torch.Tensor] = None,
                 list_target_geometry: List[torch.FloatTensor] = None,
                 list_rdkit_geometry: List[torch.FloatTensor] = None,
                 extra_dict_keys: List[str] = None):
        super(GeoMolDataset, self).__init__()
        self.n_pack = len(list_packed_mol_graph)
        assert len(list_smiles_set) == self.n_pack
        if list_properties is not None:
            assert len(list_properties) == self.n_pack
        if list_target_geometry is not None:
            assert len(list_target_geometry) == self.n_pack
        if list_rdkit_geometry is not None:
            assert len(list_rdkit_geometry) == self.n_pack

        self.list_packed_mol_graph = list_packed_mol_graph
        self.list_smiles_set = list_smiles_set
        self.list_properties = list_properties
        self.list_target_geometry = list_target_geometry
        self.list_rdkit_geometry = list_rdkit_geometry
        self.list_extra_dict: List[Dict[str, Any]] = [{} for _ in range(self.n_pack)]
        if extra_dict_keys is None:
            extra_dict_keys = []

    def __getitem__(self, index) -> Tuple[PackedMolGraph, List[str], torch.Tensor,
                                          torch.FloatTensor, torch.FloatTensor, Dict[str, Any]]:
        return self.list_packed_mol_graph[index], self.list_smiles_set[index], \
               self.list_properties[index] if self.list_properties is not None else None, \
               self.list_target_geometry[index] if self.list_target_geometry is not None else None, \
               self.list_rdkit_geometry[index] if self.list_rdkit_geometry is not None else None, \
               self.list_extra_dict[index]

    def __len__(self):
        return self.n_pack


PICKLE_DIR = 'data/pickles'
LOAD_FUNCTION_DICT = {
    SupportedDataset.QM7: load_qm7,
    SupportedDataset.LIPOP: load_lipop,
    SupportedDataset.FREESOLV: load_freesolv,
    SupportedDataset.ESOL: load_esol,
    SupportedDataset.TOX21: load_tox21,
    SupportedDataset.SARS: load_sars,
}
DATASET_CONTAINS_TARGET_GEOMETRY = [
    SupportedDataset.QM7
]
REGRESSION_DATASET = [
    SupportedDataset.QM7,
    SupportedDataset.LIPOP,
    SupportedDataset.FREESOLV,
    SupportedDataset.ESOL,
]
MULTI_CLASSIFICATION_DATASET = {
    SupportedDataset.TOX21: 2,
    SupportedDataset.SARS: 4,
}


def load_data(dataset_name: SupportedDataset, split_seed: int, dataset_token: str = None, n_mol_per_pack: int = 1,
              max_num=-1, force_save=False,
              use_tqdm=False
              ) -> Tuple[GeoMolDataset, GeoMolDataset, GeoMolDataset, Dict[str, Any]]:
    make_dir(PICKLE_DIR)
    if dataset_token is None:
        pickle_title = f'{dataset_name}-{split_seed}'
    else:
        pickle_title = f'{dataset_name}-{dataset_token}-{split_seed}'
    pickle_path = f'{PICKLE_DIR}/{pickle_title}.pickle'
    if not force_save:
        try:
            with open(pickle_path, 'rb') as fp:
                train_set, validate_set, test_set, info_dict = pickle.load(fp)
            return train_set, validate_set, test_set, info_dict
        except FileNotFoundError:
            pass
        except EOFError:
            pass

    # dump data
    molecules, properties = LOAD_FUNCTION_DICT[dataset_name](max_num, force_save)
    split_path = f'data/{dataset_name}/split-{split_seed}.json'
    if os.path.exists(split_path):
        generate_split(LOAD_FUNCTION_DICT[dataset_name], split_seed, f'data/{dataset_name}')
    with open(split_path) as fp:
        split_dict = json.load(fp)

    info_dict = {'target_dim': properties.shape[1]}
    if properties is not None:
        if dataset_name in REGRESSION_DATASET:
            # label normalization for regression
            properties = torch.FloatTensor(properties)
            mean, std = get_mean_std(torch.FloatTensor(properties[split_dict['train'], :]))
            properties = (properties - mean) / std
            print(f'\t\tmean: {mean}, std: {std}')
            info_dict['mean'], info_dict['std'] = mean, std

        if dataset_name in MULTI_CLASSIFICATION_DATASET.keys():
            # label normalization for multi-classification
            n_label = properties.shape[1]
            n_class = MULTI_CLASSIFICATION_DATASET[dataset_name]
            cnt_notnan = []
            cnt_label_class = np.ones(shape=[n_label, n_class], dtype=np.float)
            for i in range(n_label):
                labels_i = properties[:, i]
                labels_i = labels_i[np.logical_not(np.isnan(labels_i))]
                cnt_notnan.append(len(labels_i))
                n_class_b = len(set(labels_i))
                # assert n_class == n_class_b, f'{n_class} vs {n_class_b}'
                for label in labels_i:
                    cnt_label_class[i][int(label)] += 1.
            weight_label_class = (np.repeat(np.expand_dims(cnt_notnan, -1), n_class,
                                            axis=-1) / n_class) * cnt_label_class ** -1
            print(f'\t\tLabel-Class: \n{cnt_label_class}')
            print(f'\t\tWeights: \n{weight_label_class}')
            info_dict['weight_label_class'] = weight_label_class

    dataset_dict = {}
    for split_token, split_seq in split_dict.items():
        print(f'\t\tGenerating {split_token} set...')
        indices_each_pack = split_by_interval(len(split_seq), n_mol_per_pack, given_list=split_seq)
        list_packed_mol_graph = []
        list_smiles_set = []
        list_properties = []
        list_target_geometry = []
        list_rdkit_geometry = []
        if use_tqdm:
            indices_each_pack = tqdm.tqdm(indices_each_pack, total=len(indices_each_pack))
        for indices in indices_each_pack:
            packed_mol_graph = PackedMolGraph([molecules[idx] for idx in indices])
            if packed_mol_graph.n_mol == 0:
                continue
            indices = [indices[i] for i in packed_mol_graph.mask]
            list_packed_mol_graph.append(packed_mol_graph)
            list_smiles_set.append([Chem.MolToSmiles(molecules[idx]) for idx in indices])
            if properties is not None:
                list_properties.append(properties[indices, :])
            if dataset_name in DATASET_CONTAINS_TARGET_GEOMETRY:
                list_target_geometry.append(
                    torch.FloatTensor(np.vstack([target_positions(molecules[idx]) for idx in indices]))
                )
            list_rdkit_geometry.append(
                torch.FloatTensor(np.vstack([rdkit_positions(molecules[idx]) for idx in indices]))
            )
        dataset_dict[split_token] = GeoMolDataset(
            list_packed_mol_graph, list_smiles_set,
            list_properties if len(list_properties) else None,
            list_target_geometry if len(list_target_geometry) else None,
            list_rdkit_geometry if len(list_rdkit_geometry) else None
        )

    with open(pickle_path, 'wb+') as fp:
        pickle.dump((dataset_dict['train'], dataset_dict['validate'], dataset_dict['test'], info_dict), fp)

    return dataset_dict['train'], dataset_dict['validate'], dataset_dict['test'], info_dict
