import shutil
from itertools import chain

import numpy as np

from open_kbp import DataLoader
from utils import DATA_DIR


def clean_data():
    """
    Remove patients that are missing 2 or more of the organs-at-risk that were used in the original manuscript.
    """
    training_data_dir = DATA_DIR / "train-pats"
    validation_data_dir = DATA_DIR / "validation-pats"
    testing_data_dir = DATA_DIR / "test-pats"
    experiment_oars = {"Larynx", "RightParotid", "LeftParotid", "Mandible"}
    all_data = chain(training_data_dir.iterdir(), validation_data_dir.iterdir(), testing_data_dir.iterdir())
    for path in all_data:
        all_data_names = {pt_data_path.stem for pt_data_path in path.iterdir()}
        num_oars = sum(1 for oar in experiment_oars if oar in all_data_names)
        if num_oars < 3:
            print(f"Removing {path}.")
            shutil.rmtree(path)
