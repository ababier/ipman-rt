from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from algorithms.base import Algorithm
from algorithms.evaluation.summary import Summary
from networks import Generator
from numpy.typing import NDArray
from open_kbp import DataBatch, DataLoader
from oracle import Oracle
from results_manager import ResultsManager


class Evaluation(Algorithm):
    def __init__(
        self, data_loader: DataLoader, oracle: Oracle, results_manager: ResultsManager, include_validation: bool = False, include_test: bool = False
    ):
        super().__init__(data_loader, oracle, results_manager)
        self.include_validation = include_validation
        self.include_test = include_test

    def _run_iteration(self, model_name: str, lambda_: Optional[int], iteration: Optional[int]):
        self.results_manager.set_state(model_name, lambda_, iteration)
        generator = Generator.read(self.data_loader.data_shapes, self.experiment_id, model_name, lambda_, iteration, epoch=-1)
        summary = Summary(pd.DataFrame())
        if self.include_validation:
            validation_data_loader = DataLoader("validation")
            summary.add(self._evaluate_data_loader(generator, validation_data_loader).values)
        if self.include_test:
            test_data_loader = DataLoader("test")
            summary.add(self._evaluate_data_loader(generator, test_data_loader).values)
        self.results_manager.evaluation.write(summary.values)

    def _evaluate_data_loader(self, generator: Generator, data_loader: DataLoader) -> Summary:
        data_loader.batch_size = 1
        summary = Summary(pd.DataFrame())
        for batch in data_loader.get_batches():
            batch = self.normalize_batch_dose(batch)
            generated_samples = generator.predict(batch)
            criteria_evaluated = self._get_summary_statistics(batch, generated_samples, data_loader.dataset_name)
            summary.add(criteria_evaluated)
        return summary

    def _calculate_objective(self, batch: DataBatch, dose: NDArray):
        mean_oar_doses = []
        for roi_name, batch_roi_mask in batch.yield_roi_labels_and_masks(get_full_ptvs=True):
            if roi_name in self.data_loader.rois["oars"]:
                mean_doses = np.array(
                    [dose[np.where(roi_mask)].mean() if np.any(roi_mask) else np.nan for roi_mask, dose in zip(batch_roi_mask, dose)]
                )
                mean_oar_doses.append(mean_doses)
        return np.nanmean(mean_oar_doses, axis=0)

    def _evaluate_known_constraints(self, batch: DataBatch, dose: NDArray) -> dict[str, NDArray[float]]:
        known_constraints = {}
        for roi_name, batch_roi_mask in batch.yield_roi_labels_and_masks(get_full_ptvs=True):
            mean_doses = np.array([dose[np.where(roi_mask)].mean() if np.any(roi_mask) else np.nan for roi_mask, dose in zip(batch_roi_mask, dose)])
            known_constraints[f"{roi_name}_lower_known_constraint"] = mean_doses - self.historical_plan_bounds.mean_dose_limits_lb[roi_name]
            known_constraints[f"{roi_name}_upper_known_constraint"] = self.historical_plan_bounds.mean_dose_limits_ub[roi_name] - mean_doses
        return known_constraints

    def _evaluate_hidden_constraints(self, batch: DataBatch, generated_dose: NDArray) -> dict[str, NDArray[bool]]:
        batch_dose = self.historical_plan_bounds.get_dose_in_gray(batch.dose)
        batch_criteria = self.oracle.evaluate_criteria(batch_dose, batch, is_ref=True)
        generated_criteria = self.oracle.evaluate_criteria(generated_dose, batch)
        relaxed_generated_criteria = self.oracle.evaluate_criteria(generated_dose, batch, tolerance=1)

        hidden_constraints = {}
        constraint_values = np.where(batch_criteria > 0, generated_criteria, np.nan)
        relaxed_constraint_values = np.where(batch_criteria > 0, relaxed_generated_criteria, np.nan)
        for index, roi_name in enumerate(batch.structure_mask_names):
            hidden_constraints[f"{roi_name}_context_constraint"] = constraint_values[:, index]
            hidden_constraints[f"{roi_name}_context_constraint_relaxed"] = relaxed_constraint_values[:, index]
        return hidden_constraints

    def _get_summary_statistics(self, batch: DataBatch, generated_dose: NDArray, dataset_name: str) -> pd.DataFrame:
        generated_dose_in_gy = self.historical_plan_bounds.get_dose_in_gray(generated_dose)
        metrics = defaultdict(lambda: np.full(batch.size, np.nan))
        metrics["patient_id"] = np.array(batch.patient_list)
        metrics["dataset"] = np.full(batch.size, dataset_name)
        metrics["objective"] = self._calculate_objective(batch, generated_dose_in_gy)
        metrics["feasibility"] = self.oracle.label_dose_samples(generated_dose, batch).all(axis=1).squeeze().astype(int)
        metrics |= self._evaluate_hidden_constraints(batch, generated_dose_in_gy)
        metrics |= self._evaluate_known_constraints(batch, generated_dose_in_gy)
        metrics_df = pd.DataFrame(metrics)
        return metrics_df.set_index("patient_id")
