import os
import json
import pickle
import numpy as np
import pandas as pd
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem.rdchem import Mol as Molecule
from typing import Tuple, List

from data.config import ESOL_CSV_PATH, ESOL_PICKLE_PATH


def dump_esol():
    df = pd.read_csv(ESOL_CSV_PATH)
    csv: np.ndarray = df.values
    smiles = csv[:, 9].astype(np.str)
    properties = csv[:, 8: 9].astype(np.float)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    with open(ESOL_PICKLE_PATH, 'wb+') as fp:
        pickle.dump((mols, properties), fp)


def load_esol(max_num=-1, force_save=False) -> Tuple[List[Molecule], np.ndarray]:
    if not os.path.exists(ESOL_PICKLE_PATH) or force_save:
        dump_esol()
    with open(ESOL_PICKLE_PATH, 'rb') as fp:
        mols, properties = pickle.load(fp)
    if 0 < max_num < len(mols):
        mols = mols[: max_num]
        properties = properties[: max_num, :]
    return mols, properties
