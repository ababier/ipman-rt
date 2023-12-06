import numpy as np
from numpy.typing import NDArray

from open_kbp import DataBatch
from utils.historical_plan_bounds import HistoricalPlanBounds


class Oracle:
    """
    Creates labels of 0 or 1 if a constraint is failed or satisfied, respectively.
    Missing structures are labeled with an `np.nan`.
    """

    def __init__(self, historical_plan_bounds: HistoricalPlanBounds, alternative_criteria=False):
        self.historical_plan_bounds = historical_plan_bounds
        self.alternative_criteria = alternative_criteria

    def label_dose_samples(self, dose_samples: NDArray[float], batch: DataBatch) -> NDArray:
        reference_dose_gy = self.historical_plan_bounds.get_dose_in_gray(batch.dose)
        reference_evaluation = self.evaluate_criteria(reference_dose_gy, batch, is_ref=True)

        dose_samples_gy = self.historical_plan_bounds.get_dose_in_gray(dose_samples)
        samples_evaluation = self.evaluate_criteria(dose_samples_gy, batch)
        worst_linear_pass_margins = self.evaluate_mean_constraints(dose_samples_gy, batch)

        reference_criteria_satisfied = np.where(np.isnan(reference_evaluation), np.nan, 0 <= reference_evaluation)
        sample_criteria_satisfied = np.where(np.isnan(samples_evaluation), np.nan, 0 <= samples_evaluation)
        sample_mean_satisfied = np.where(np.isnan(worst_linear_pass_margins), np.nan, 0 <= worst_linear_pass_margins)
        roi_labels = np.multiply((reference_criteria_satisfied <= sample_criteria_satisfied), sample_mean_satisfied)
        return np.expand_dims(np.array(roi_labels), 2)

    def evaluate_criteria(self, sample_doses_in_gy: NDArray[float], batch: DataBatch, tolerance: int = 0, is_ref: bool = False) -> NDArray[float]:
        """
        Evaluates clinical criteria for all ROIs.

        Args:
            sample_doses_in_gy: The dose to evaluate in units of Gy.
            batch: the batch data that is related to the sample_doses_in_gy.
            tolerance: how much we can violate a constraint before it is labeled as a fail.
            is_ref: forces the oracle to use the original criteria

        Returns:
            Matrix of labels. Each row is a batch, each column corresponds to an ROI.
        """
        criteria_pass_margins = []
        for roi_name, roi_doses in batch.yield_roi_dose(sample_doses_in_gy):
            if roi_name in {"LeftParotid", "RightParotid"}:
                criteria_pass_margins.append([26 - np.mean(dose) + tolerance for dose in roi_doses])
            elif roi_name == "Larynx":
                criteria_pass_margins.append([45 - np.mean(dose) + tolerance for dose in roi_doses])
            elif roi_name == "Mandible":
                criteria_pass_margins.append([73.5 - np.max(dose) + tolerance for dose in roi_doses])
            elif "PTV" in roi_name:
                if self.alternative_criteria and roi_name == "PTV56" and not is_ref:
                    dose_prescribed = 54
                elif self.alternative_criteria and roi_name == "PTV63" and not is_ref:
                    dose_prescribed = 66
                elif self.alternative_criteria and roi_name == "PTV70" and not is_ref:
                    dose_prescribed = 72
                else:
                    dose_prescribed = int(roi_name[3:])
                criteria_pass_margins.append([np.percentile(dose, 10) - dose_prescribed + tolerance for dose in roi_doses])
        return np.array(criteria_pass_margins).T

    def evaluate_mean_constraints(self, dose: NDArray[float], batch: DataBatch) -> NDArray[float]:
        """Evaluate the mean dose constraints (i.e., known constraints)."""
        worst_pass_margin = []
        for roi_name, roi_doses in batch.yield_roi_dose(dose):
            mean_roi_doses = np.array([dose.mean() for dose in roi_doses])
            lb_pass_margins = mean_roi_doses - self.historical_plan_bounds.mean_dose_limits_lb[roi_name]
            ub_pass_margins = self.historical_plan_bounds.mean_dose_limits_ub[roi_name] - mean_roi_doses
            worst_pass_margin.append(np.minimum(lb_pass_margins, ub_pass_margins))
        return np.array(worst_pass_margin).T
