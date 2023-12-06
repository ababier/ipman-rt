import os
from ctypes import c_void_p, c_wchar_p
from pathlib import Path
from typing import Optional

import pandas as pd
from networks.base import ModelWrapper

from results_manager.base import BaseStructure


class Loss(BaseStructure):
    def __init__(self, experiment_id: str, model_name: c_wchar_p, lambda_: c_void_p, iteration: c_void_p):
        super().__init__(experiment_id, results_name="loss", model_name=model_name, lambda_=lambda_, iteration=iteration)
        self._cache = []
        self._model: Optional[ModelWrapper] = None

    @property
    def model(self) -> ModelWrapper:
        if self._model is None:
            raise ValueError("Model has not been passed to the Loss section of the results manager.")
        return self._model

    def get_save_path(self, epoch: int) -> Path:
        return self.parent_dir.parent / self.model.name / self.lambda_dir / self.iteration_dir / f"epoch_{epoch}.csv"

    def set_new_model(self, model: ModelWrapper) -> None:
        self._model = model
        self._cache = []

    def log_loss(self, loss: float | list[float]) -> None:
        loss_list = loss if isinstance(loss, list) else [loss]
        self._cache.append(loss_list)

    def write_and_flush_cache(self, epoch: int):
        save_path = self.get_save_path(epoch)
        cache_df = pd.DataFrame(data=self._cache, columns=[self.model.loss_names])
        os.makedirs(save_path.parent, exist_ok=True)
        cache_df.to_csv(save_path)
        self._cache = []

    def read_all(self) -> pd.DataFrame:
        df = pd.DataFrame()
        for _, iteration, iterations_dir in self._get_iteration_samples():
            lambda_value = None if "lambda" not in iterations_dir.parent.stem else int(iterations_dir.parent.stem.split("lambda_")[-1])
            epochs_by_path = sorted({int(cache_file.stem.split("epoch_")[-1]): cache_file for cache_file in iterations_dir.iterdir()}.items())
            for epoch_num, cache_file in epochs_by_path:
                df_tmp = pd.read_csv(cache_file)
                df_tmp["lambda_"] = lambda_value
                df_tmp["iteration"] = iteration
                df_tmp["epoch"] = epoch_num
                df = pd.concat((df, df_tmp))
        return df
