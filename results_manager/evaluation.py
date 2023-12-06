import os
from ctypes import c_void_p, c_wchar_p
from pathlib import Path

import pandas as pd
from algorithms.evaluation.summary import Summary

from results_manager.base import BaseStructure


class Evaluation(BaseStructure):
    def __init__(self, experiment_id: str, model_name: c_wchar_p, lambda_: c_void_p, iteration: c_void_p):
        super().__init__(experiment_id, "evaluation", model_name, lambda_, iteration)

    @property
    def state_path(self) -> Path:
        return self.parent_dir / self.lambda_dir / f"iteration_{self.iteration}.csv"

    def write(self, metrics: pd.DataFrame) -> None:
        save_path = self.state_path
        if save_path.exists():
            previous_metrics = pd.read_csv(save_path, index_col="patient_id")
            patient_intersection = previous_metrics.index.intersection(metrics.index)
            metrics = pd.concat((metrics, previous_metrics.drop(patient_intersection)))
        else:
            os.makedirs(save_path.parent, exist_ok=True)
        metrics.to_csv(save_path)

    def read_all(self):
        df = pd.DataFrame()
        for model_name, iteration, iteration_path in self._get_iteration_samples(get_all=True):
            df_tmp = pd.read_csv(iteration_path)
            df_tmp["model_name"] = model_name
            df_tmp["lambda_"] = None if "lambda" not in iteration_path.parent.stem else int(iteration_path.parent.stem.split("lambda_")[-1])
            df_tmp["iteration"] = iteration
            df = pd.concat((df, df_tmp))
        return Summary(df)


