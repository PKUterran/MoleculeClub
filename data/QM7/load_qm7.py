import os
import pickle
import numpy as np
import pandas as pd
import rdkit.Chem as Chem
from typing import Tuple, List
from rdkit.Chem.rdchem import Mol as Molecule
from data.config import QM7_CSV_PATH, QM7_SDF_PATH, QM7_PICKLE_PATH


def dump_qm7():
    supplier = Chem.SDMolSupplier(f'data/{QM7_SDF_PATH}')
    mols = [m for m in supplier if m is not None and m.GetProp("_Name").startswith("gdb7k")]
    mols = [Chem.RemoveAllHs(mol) for mol in mols]
    indices = [int(m.GetProp("_Name")[6: 10]) for m in mols]
    df = pd.read_csv(f'data/{QM7_CSV_PATH}')
    csv: np.ndarray = df.values
    properties = csv[:, 0: 1].astype(np.float)
    properties = properties[indices, :]
    with open(f'data/{QM7_PICKLE_PATH}', 'wb+') as fp:
        pickle.dump((mols, properties), fp)


def load_qm7(max_num=-1, force_save=False) -> Tuple[List[Molecule], np.ndarray]:
    if not os.path.exists(f'data/{QM7_PICKLE_PATH}') or force_save:
        dump_qm7()
    with open(f'data/{QM7_PICKLE_PATH}', 'rb') as fp:
        mols, properties = pickle.load(fp)
    if 0 < max_num < len(mols):
        mols = mols[: max_num]
        properties = properties[: max_num, :]
    return mols, properties
