import os
import json
import pickle
import numpy as np
import pandas as pd
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem.rdchem import Mol as Molecule
from typing import Tuple, List

from data.config import TOX21_CSV_PATH, TOX21_PICKLE_PATH


def dump_tox21():
    df = pd.read_csv(TOX21_CSV_PATH)
    csv: np.ndarray = df.values
    smiles = csv[:, 13].astype(np.str)
    properties = csv[:, : 12].astype(np.float)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    with open(TOX21_PICKLE_PATH, 'wb+') as fp:
        pickle.dump((mols, properties), fp)


def load_tox21(max_num=-1, force_save=False) -> Tuple[List[Molecule], np.ndarray]:
    if not os.path.exists(TOX21_PICKLE_PATH) or force_save:
        dump_tox21()
    with open(TOX21_PICKLE_PATH, 'rb') as fp:
        mols, properties = pickle.load(fp)
    if 0 < max_num < len(mols):
        mols = mols[: max_num]
        properties = properties[: max_num, :]
    return mols, properties
