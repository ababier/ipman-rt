import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from open_kbp import DataLoader

from utils.global_directories import EXPERIMENTS_DIR


@dataclass
class HistoricalPlanBounds:
    max_dose: int
    mean_dose_limits_lb: dict[str, int]
    mean_dose_limits_ub: dict[str, int]

    def get_dose_in_gray(self, dose: NDArray[float]):
        return (dose + 1) / 2 * self.max_dose

    def get_normalized_dose(self, dose_in_gray: NDArray[float]):
        return dose_in_gray / self.max_dose * 2 - 1

    def get_normalized_mean_ub(self, roi_name: str):
        return self.mean_dose_limits_ub[roi_name]

    def get_normalized_mean_lb(self, roi_name: str):
        return self.mean_dose_limits_lb[roi_name]

    def make_alternate_criteria(self):
        self.mean_dose_limits_ub["PTV70"] += 2
        self.mean_dose_limits_ub["PTV63"] += 3
        self.mean_dose_limits_ub["PTV56"] -= 2

    @classmethod
    def get(cls, data_loader: DataLoader, use_alternative_criteria: bool = False, recalculate: bool = False):
        data_path = EXPERIMENTS_DIR / "historical_plan_bounds.csv"
        if data_path.exists() and not recalculate:
            historical_plan_bounds = cls.read(data_path)
        else:
            historical_plan_bounds = cls.from_dataset(data_loader)
            historical_plan_bounds.write(data_path)

        if use_alternative_criteria:
            historical_plan_bounds.make_alternate_criteria()

        return historical_plan_bounds

    @classmethod
    def read(cls, data_path: Path):
        with open(data_path, "r") as fd:
            return cls(**json.load(fd))

    def write(self, data_path: Path):
        os.makedirs(data_path.parent, exist_ok=True)
        with open(data_path, "w") as fd:
            json.dump(asdict(self), fd)

    @classmethod
    def from_dataset(cls, data_loader: DataLoader):
        mean_dose_limits_lb = defaultdict(lambda: float("inf"))
        mean_dose_limits_ub = defaultdict(lambda: float("-inf"))
        max_dose = float("-inf")
        data_loader.set_files_to_load("primary")
        data_loader.batch_size = 1
        for batch in data_loader.get_batches():
            max_dose = max(max_dose, np.max(batch.dose))
            for roi_name, batch_roi_mask in batch.yield_roi_labels_and_masks(get_full_ptvs=True):
                mean_roi_doses = np.array(
                    [dose[np.where(roi_mask)].mean() if np.any(roi_mask) else np.nan for roi_mask, dose in zip(batch_roi_mask, batch.dose)]
                )
                mean_dose_limits_lb[roi_name] = min(mean_dose_limits_lb[roi_name], np.min(mean_roi_doses))
                mean_dose_limits_ub[roi_name] = max(mean_dose_limits_ub[roi_name], np.max(mean_roi_doses))

        # Round bounds to the nearest and least strict integer bound
        rounded_mean_dose_limits_lb = {roi_name: np.floor(lb) for roi_name, lb in mean_dose_limits_lb.items()}
        rounded_mean_dose_limits_ub = {roi_name: np.ceil(ub) for roi_name, ub in mean_dose_limits_ub.items()}

        return cls(np.ceil(max_dose), rounded_mean_dose_limits_lb, rounded_mean_dose_limits_ub)
