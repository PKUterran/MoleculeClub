import numpy as np
from typing import Union, List, Tuple, Dict
from rdkit import Chem


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


ATOMS_MASS = {
    'B': 10.811,
    'C': 12.0107,
    'N': 14.0067,
    'O': 15.9994,
    'F': 18.9984032,
    'Si': 28.2855,
    'P': 30.973762,
    'S': 32.065,
    'Cl': 35.453,
    'As': 74.92160,
    'Se': 78.96,
    'Br': 79.904,
    'Te': 127.60,
    'I': 126.90447,
    'At': 209.9871,
    'other': 50.0,
}

ATOMS = list(ATOMS_MASS.keys())


def get_atoms_massive_matrix(atoms: list) -> np.ndarray:
    massive = []
    for a in atoms:
        massive.append(ATOMS_MASS[a])
    massive = np.vstack([np.array(massive).reshape([-1, 1]), np.zeros([num_atom_features() - len(atoms), 1])])
    return massive


def get_default_atoms_massive_matrix() -> np.ndarray:
    return get_atoms_massive_matrix(ATOMS)


def get_massive_from_atom_features(af: np.ndarray) -> np.ndarray:
    return np.asmatrix(af) @ np.asmatrix(get_default_atoms_massive_matrix()) / 50


def atom_features(atom,
                  explicit_H=True,
                  use_chirality=True,
                  default_atoms=None):
    if not default_atoms:
        default_atoms = ATOMS
    results = \
        one_of_k_encoding_unk(atom.GetSymbol(), default_atoms) + \
        one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) + \
        [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
        one_of_k_encoding_unk(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2, 'other'
        ]) + \
        [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9-small), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]

    return np.array(results, dtype=np.int)


def bond_features(bond):
    use_chirality = True
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return np.array(bond_feats, dtype=np.int)


def num_atom_features():
    # Return length of feature vector using a very simple molecule.
    m = Chem.MolFromSmiles('CC')
    alist = m.GetAtoms()
    a = alist[0]
    return len(atom_features(a))


def num_bond_features():
    # Return length of feature vector using a very simple molecule.
    simple_mol = Chem.MolFromSmiles('CC')
    Chem.SanitizeMol(simple_mol)
    return len(bond_features(simple_mol.GetBonds()[0]))


def encode_smiles(smiles: np.ndarray) -> Tuple[List[Dict[str, np.ndarray]], List[int]]:
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    return encode_mols(mols)


def encode_mols(mols: list) -> Tuple[List[Dict[str, np.ndarray]], List[int]]:
    ret = []
    mask = []
    # print('\tStart encoding...')
    cnt = 0
    for idx, mol in enumerate(mols):
        info = {}
        if not mol:
            continue
        else:
            mask.append(idx)
        info['af'] = np.stack([atom_features(a) for i, a in enumerate(mol.GetAtoms())])

        info['bf'] = np.stack([bond_features(b) for b in mol.GetBonds()]
                              # + [bond_features(b) for b in mol.GetBonds()]
                              ) if len(mol.GetBonds()) else np.zeros(shape=[0, 10], dtype=np.int)
        info['us'] = np.array([b.GetBeginAtomIdx() for b in mol.GetBonds()]
                              # + [b.GetEndAtomIdx() for b in mol.GetBonds()]
                              , dtype=np.int)
        info['vs'] = np.array([b.GetEndAtomIdx() for b in mol.GetBonds()]
                              # + [b.GetBeginAtomIdx() for b in mol.GetBonds()]
                              , dtype=np.int)
        ret.append(info)
        cnt += 1
        # if cnt % 10000 == 0:
        #     print('\t', cnt, 'encoded.')
    # print(f'\tEncoded: {cnt}/{len(mols)}')
    return ret, mask


def get_features_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    af = np.stack([atom_features(a) for i, a in enumerate(mol.GetAtoms())])
    bf = np.stack([bond_features(b) for b in mol.GetBonds()]) \
        if len(mol.GetBonds()) else np.zeros(shape=[0, 10], dtype=np.int)
    us = np.array([b.GetBeginAtomIdx() for b in mol.GetBonds()], dtype=np.int)
    vs = np.array([b.GetEndAtomIdx() for b in mol.GetBonds()], dtype=np.int)
    return af, bf, us, vs
