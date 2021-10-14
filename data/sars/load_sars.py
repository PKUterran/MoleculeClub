import os
import json
import pickle
import numpy as np
import pandas as pd
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem.rdchem import Mol as Molecule
from typing import Tuple, List

from data.config import SARS_CSV_PATH, SARS_PICKLE_PATH


def dump_sars():
    df = pd.read_csv(SARS_CSV_PATH)
    csv: np.ndarray = df.values
    smiles = csv[:, 0].astype(np.str)
    properties = csv[:, 1: 14].astype(np.float)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    mask = [i for i, m in enumerate(mols) if m is not None]
    mols = [mols[i] for i in mask]
    properties = properties[mask, :]
    with open(SARS_PICKLE_PATH, 'wb+') as fp:
        pickle.dump((mols, properties), fp)


def load_sars(max_num=-1, force_save=False) -> Tuple[List[Molecule], np.ndarray]:
    if not os.path.exists(SARS_PICKLE_PATH) or force_save:
        dump_sars()
    with open(SARS_PICKLE_PATH, 'rb') as fp:
        mols, properties = pickle.load(fp)
    if 0 < max_num < len(mols):
        mols = mols[: max_num]
        properties = properties[: max_num, :]
    return mols, properties
