import json
import os

LOG_DIR = 'log'


def save_log(log, directory: str, tag: str):
    if not os.path.exists(f'{LOG_DIR}/{directory}'):
        os.mkdir(f'{LOG_DIR}/{directory}')
    path = f'{LOG_DIR}/{directory}/{tag}.json'
    with open(path, 'w+') as fp:
        json.dump(log, fp)
