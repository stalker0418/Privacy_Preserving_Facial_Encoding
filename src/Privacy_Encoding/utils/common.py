from ensure import ensure_annotations
from box import Box, ConfigBox
from pathlib import Path
from typing import List

import yaml
from src.Privacy_Encoding.logging import logger
from box.exceptions import BoxValueError
import numpy as np
import cv2
import os




@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


def normalize_dataset(dataset):
    # Assuming 'dataset' is a 3D NumPy array representing images
    # Normalize to the range [0, 1]
    normalized_dataset = (dataset.astype(np.float32) / 255.0).clip(0.0, 1.0)

    # Alternatively, normalize to the range [-1, 1]
    # normalized_dataset = (dataset.astype(np.float32) / 127.5) - 1.0

    return normalized_dataset


@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"



@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")




