from ctypes import c_void_p, c_wchar_p
from pathlib import Path
from typing import Iterator, Optional

from utils import EXPERIMENTS_DIR


class BaseStructure:
    def __init__(self, experiment_id: str, results_name: str, model_name: c_wchar_p, lambda_: c_void_p, iteration: c_void_p):
        self.results_name = results_name
        self.experiment_id = experiment_id
        self._model_name = model_name
        self._lambda = lambda_
        self._iteration = iteration

    @property
    def model_name(self) -> Optional[int]:
        return self._model_name.value

    @property
    def lambda_(self) -> Optional[int]:
        return self._lambda.value

    @property
    def iteration(self) -> Optional[int]:
        return self._iteration.value

    @property
    def _parent_dir(self) -> Path:
        return EXPERIMENTS_DIR / self.experiment_id / self.results_name

    @property
    def parent_dir(self) -> Path:
        return EXPERIMENTS_DIR / self.experiment_id / self.results_name / (self.model_name or "")

    @property
    def lambda_dir(self) -> str:
        return f"lambda_{self.lambda_}" if self.lambda_ is not None else ""

    @property
    def iteration_dir(self) -> str:
        return f"iteration_{self.iteration}" if self.iteration is not None else ""

    def _get_iteration_samples(self, get_all: bool = False) -> Iterator[tuple[str, int, Path]]:
        for model_dir in self._parent_dir.iterdir() if get_all else [self.parent_dir]:
            if any("lambda_" in dir_.stem for dir_ in model_dir.iterdir()):
                for lambda_dir in model_dir.iterdir():
                    for iteration_path in lambda_dir.iterdir():
                        iteration = iteration_path.stem.split("_")[-1]
                        iteration = 0 if iteration == "None" else int(iteration)
                        yield model_dir.name, iteration, iteration_path
            elif any("iteration_" in dir_.stem for dir_ in model_dir.iterdir()):
                for iteration_path in model_dir.iterdir():
                    iteration = iteration_path.stem.split("_")[-1]
                    iteration = 1 if iteration == "None" else int(iteration)
                    yield model_dir.name, iteration, iteration_path
            else:
                iteration = 1
                yield model_dir.name, iteration, model_dir
