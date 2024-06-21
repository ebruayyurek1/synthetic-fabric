import json
from pathlib import Path

import yaml
from tqdm import tqdm


def load_yaml(path: str | Path) -> dict:
    """
    Load YAML as python dict

    @param path: path to YAML file
    @return: dictionary containing data
    """
    with open(path, encoding="UTF-8") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    return data


def dump_yaml(data, path: str | Path) -> None:
    """
    Load YAML as python dict

    @param path: path to YAML file
    @param data: data to dump
    @return: dictionary containing data
    """
    with open(path, encoding="UTF-8", mode="w") as f:
        yaml.dump(data, f, Dumper=yaml.SafeDumper)



def read_json_files(file_paths):
    # Function to read JSON files and yield dictionaries
    for file_path in tqdm(file_paths):
        with open(file_path, 'r') as f:
            yield json.load(f)
