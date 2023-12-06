import random
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Iterator, List, Union

import numpy as np
from more_itertools import peekable, windowed
from numpy.typing import NDArray
from tqdm import tqdm

from open_kbp.batch import DataBatch
from open_kbp.data_shapes import DataShapes
from open_kbp.utils import get_paths, load_file
from utils import DATA_DIR


class DatasetName(Enum):
    train = "train"
    validation = "validation"
    test = "test"

    def __str__(self) -> str:
        return self.value


class DataLoader:
    """Loads OpenKBP csv data in structured format for dose prediction models."""

    def __init__(self, dataset_name: str = "train", batch_size: int = 2):
        self.dataset_name = DatasetName(dataset_name)
        self.batch_size = batch_size

        # Parameters that should not be changed unless OpenKBP data is modified
        self.rois = dict(
            oars=["RightParotid", "LeftParotid", "Larynx", "Mandible"],  # Removed "Brainstem", "SpinalCord", "Esophagus"
            targets=["PTV56", "PTV63", "PTV70"],
        )
        self.full_roi_list = sum(map(list, self.rois.values()), [])  # make a list of all rois
        self.num_rois = len(self.full_roi_list)
        self.data_shapes = DataShapes(self.rois)

        # Dependent on input and OpenKBP parameters above
        self.data_dir = DATA_DIR / f"{self.dataset_name}-pats"
        self.patient_paths = list(self.data_dir.iterdir())
        self._required_files = None
        self.set_files_to_load()
        self.dose_sample_paths: list[Path] = []

    @property
    def paths_by_patient_id(self) -> dict[str, Path]:
        return {patient_path.stem: patient_path for patient_path in self.patient_paths}

    @property
    def num_batches(self) -> int:
        return int(np.floor(len(self.batch_paths) / self.batch_size))

    @property
    def batch_paths(self) -> list[Path]:
        batch_paths = [*self.patient_paths, *self.dose_sample_paths]
        random.shuffle(batch_paths)
        return batch_paths

    def get_batches(self) -> Iterator[DataBatch]:
        batches = windowed(self.batch_paths, n=self.batch_size, step=self.batch_size)
        complete_batches = peekable(batch for batch in batches if None not in batch)
        for batch_paths in tqdm(complete_batches, total=self.num_batches):
            batch = self.create_batch(batch_paths)
            yield batch

    def get_patients(self, patient_list: List[str]) -> DataBatch:
        file_paths_to_load = [self.paths_by_patient_id[patient] for patient in patient_list]
        return self.create_batch(file_paths_to_load)

    def set_files_to_load(self, mode: str = "primary") -> None:
        """Set parameters based on `mode`."""
        if mode == "primary":
            required_data = ["dose", "ct", "structure_masks", "possible_dose_mask", "voxel_dimensions"]
        elif mode == "with_sample":
            required_data = ["dose", "sample_dose", "ct", "structure_masks", "possible_dose_mask", "voxel_dimensions"]
        else:
            raise ValueError(f"Mode '{mode}' does not exist. Mode must be either 'primary' or 'with_sample'")
        self._required_files = self.data_shapes.from_data_names(required_data)

    def create_batch(self, file_paths_to_load: List[Path]) -> DataBatch:
        batch_data = DataBatch.initialize_from_required_data(self._required_files, self.batch_size)
        batch_data.patient_list = [data_path.stem if data_path.is_dir() else data_path.parent.stem for data_path in file_paths_to_load]
        batch_data.structure_mask_names = self.full_roi_list

        # Populate batch with requested data
        for index, file_path in enumerate(file_paths_to_load):
            patient_path = file_path if file_path.is_dir() else self.paths_by_patient_id[file_path.parent.stem]
            raw_data = self.load_data(patient_path)
            for key in self._required_files:
                if key == "sample_dose":
                    sampled_dose = {key: load_file(file_path / "dose.csv" if file_path.is_dir() else file_path)}
                    batch_data.set_values(key, index, self.prepare_data_as_tensor(key, sampled_dose))
                else:
                    batch_data.set_values(key, index, self.prepare_data_as_tensor(key, raw_data))

        return batch_data

    def load_data(self, path_to_load: Path) -> Union[NDArray, dict[str, NDArray]]:
        """Load data in its raw form."""
        data = {}
        if path_to_load.is_dir():
            files_to_load = get_paths(path_to_load)
            for file_path in files_to_load:
                is_required = file_path.stem in self._required_files
                is_required_roi = file_path.stem in self.full_roi_list
                if is_required or is_required_roi:
                    data[file_path.stem] = load_file(file_path)
        return data

    def prepare_data_as_tensor(self, key: str, data: dict) -> NDArray:
        """Prepare data in a tensor that is scaled and shape for training deep learning models"""
        tensor = np.zeros(self._required_files[key])
        if key == "structure_masks":
            for roi_idx, roi in enumerate(self.full_roi_list):
                if roi in data.keys():
                    np.put(tensor, self.num_rois * data[roi] + roi_idx, int(1))
        elif key == "possible_dose_mask":
            np.put(tensor, data[key], int(1))
        elif key == "voxel_dimensions":
            tensor = data[key]
        else:
            np.put(tensor, data[key]["indices"], data[key]["data"])

        if key == "ct":
            tensor = tensor.clip(0, 4095) / 4095 * 2 - 1

        return tensor
