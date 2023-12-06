import os
import random
from ctypes import c_void_p, c_wchar_p
from itertools import chain
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd
from more_itertools import partition
from numpy.typing import NDArray
from open_kbp.utils import sparse_vector_function

from results_manager.base import BaseStructure


class SampleDose(BaseStructure):
    def __init__(
        self,
        experiment_id: str,
        model_name: c_wchar_p,
        lambda_: c_void_p,
        iteration: c_void_p,
        history_length: Optional[int] = 6,
        iteration_sample_limit: Optional[int] = 100,
    ):
        super().__init__(experiment_id, "samples", model_name, lambda_, iteration)
        self.history_length = history_length
        self.iteration_sample_limit = iteration_sample_limit

    @property
    def generated_data_dir(self) -> Path:
        return self.parent_dir / self.lambda_dir / self.iteration_dir

    def get_fail_dose_path(self, patient_id: str, epoch_num: int) -> Path:
        return self.generated_data_dir / patient_id / f"fail_epoch_{epoch_num}.csv"

    def get_pass_dose_path(self, patient_id: str, epoch_num: int) -> Path:
        return self.generated_data_dir / patient_id / f"pass_epoch_{epoch_num}.csv"

    def write(self, sample_dose_in_gray: NDArray, is_pass: bool, patient_id: str, epoch: int) -> None:
        sparse_dose_in_gray = sparse_vector_function(sample_dose_in_gray)
        dose_df = pd.DataFrame(data=sparse_dose_in_gray["data"].squeeze(), index=sparse_dose_in_gray["indices"].squeeze(), columns=["data"])
        save_path = self.get_pass_dose_path(patient_id, epoch) if is_pass else self.get_fail_dose_path(patient_id, epoch)
        os.makedirs(save_path.parent, exist_ok=True)
        dose_df.to_csv(save_path)

    def get_recent_sample_dose_paths(self) -> list[Path]:
        recent_sample_paths = []
        for iteration_samples in self.get_recent_iteration_paths():
            iteration_samples_subset = self.get_pass_fails(iteration_samples)
            recent_sample_paths.extend(iteration_samples_subset)
        return recent_sample_paths

    def get_recent_iteration_paths(self) -> Iterator[Iterator[Path]]:
        for _, iteration_num, iteration_path in self._get_iteration_samples(get_all=True):
            if iteration_num <= self.iteration and self.iteration - iteration_num < self.history_length:
                yield chain.from_iterable(patient_path.iterdir() for patient_path in iteration_path.iterdir())

    def get_pass_fails(self, iteration_samples: Iterator[Path]) -> list[Path]:
        pass_paths_iterator, fail_paths_iterator = partition(lambda x: "fail" in x.stem, iteration_samples)
        pass_paths, fail_paths = list(pass_paths_iterator), list(fail_paths_iterator)
        num_pass_to_sample = min(len(pass_paths), self.iteration_sample_limit or float("inf"))
        num_fail_to_sample = min(len(fail_paths), self.iteration_sample_limit or float("inf"))
        return [*random.sample(pass_paths, num_pass_to_sample), *random.sample(fail_paths, num_fail_to_sample)]
