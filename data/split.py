import numpy as np
import json
from typing import Callable, List, Tuple

TRAIN_PER, VALIDATE_PER, TEST_PER = 0.8, 0.1, 0.1
assert 1 - 1e-5 < TRAIN_PER + VALIDATE_PER + TEST_PER < 1 + 1e-5


def generate_split(load_function: Callable[[], Tuple[List, np.ndarray]], seed: int, directory: str):
    mols, *_ = load_function()
    n_total = len(mols)
    print(f'directory: {directory}, seed: {seed} , # of samples: {n_total}')
    np.random.seed(seed)
    seq = np.random.permutation(n_total)
    n_train = int(n_total * TRAIN_PER)
    n_test = int(n_total * TEST_PER)
    seq_train = seq[:n_train].tolist()
    seq_validate = seq[n_train:-n_test].tolist()
    seq_test = seq[-n_test:].tolist()
    with open(f'{directory}/split-{seed}.json', 'w+') as fp:
        json.dump({
            'train': seq_train,
            'validate': seq_validate,
            'test': seq_test,
        }, fp)
