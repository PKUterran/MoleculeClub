from data.config import *
from data.utils import LIST_SEED
from data.split import generate_split
from data.Lipop.load_lipop import load_lipop
from data.FreeSolv.load_freesolv import load_freesolv
from data.ESOL.load_esol import load_esol
from data.TOX21.load_tox21 import load_tox21
from data.sars.load_sars import load_sars


if __name__ == '__main__':
    for seed in LIST_SEED:
        generate_split(load_lipop, seed, LIPOP_DIR)
        generate_split(load_freesolv, seed, FREESOLV_DIR)
        generate_split(load_esol, seed, ESOL_DIR)
        generate_split(load_tox21, seed, TOX21_DIR)
        generate_split(load_sars, seed, SARS_DIR)
