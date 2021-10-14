import os
import json
import pickle
import numpy as np
import pandas as pd
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem.rdchem import Mol as Molecule
from typing import Tuple, List

from data.config import FREESOLV_CSV_PATH, FREESOLV_PICKLE_PATH


def dump_freesolv():
    df = pd.read_csv(FREESOLV_CSV_PATH)
    csv: np.ndarray = df.values
    smiles = csv[:, 1].astype(np.str)
    properties = csv[:, 2: 3].astype(np.float)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    with open(FREESOLV_PICKLE_PATH, 'wb+') as fp:
        pickle.dump((mols, properties), fp)


def load_freesolv(max_num=-1, force_save=False) -> Tuple[List[Molecule], np.ndarray]:
    if not os.path.exists(FREESOLV_PICKLE_PATH) or force_save:
        dump_freesolv()
    with open(FREESOLV_PICKLE_PATH, 'rb') as fp:
        mols, properties = pickle.load(fp)
    if 0 < max_num < len(mols):
        mols = mols[: max_num]
        properties = properties[: max_num, :]
    return mols, properties
