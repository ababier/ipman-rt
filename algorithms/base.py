from multiprocessing import Process as IsolatedProcess

from numpy.typing import NDArray
from open_kbp import DataBatch, DataLoader
from oracle import Oracle
from results_manager import ResultsManager


class Algorithm:
    def __init__(self, data_loader: DataLoader, oracle: Oracle, results_manager: ResultsManager):
        self.data_loader = data_loader
        self.oracle = oracle
        self.results_manager = results_manager
        self.experiment_id = results_manager.experiment_id
        self.historical_plan_bounds = oracle.historical_plan_bounds

    def run(self, **kwargs) -> None:
        """
        Work around memory leak caused by loading and deleting multiple keras models.
        Args:
            **kwargs: arguments for _run_iteration, which should be defined in a child class.
        """
        process = IsolatedProcess(target=self._run_iteration, kwargs=kwargs)
        process.start()
        process.join()
        if process.exitcode:
            raise ValueError("Failed to run iteration.")

    def _run_iteration(self, **kwargs) -> None:
        raise ValueError("Not implemented. This method should be define in a child class of Algorithm")

    def normalize_batch_dose(self, batch: DataBatch) -> DataBatch:
        batch.dose = self.historical_plan_bounds.get_normalized_dose(batch.dose)
        batch.sample_dose = self.historical_plan_bounds.get_normalized_dose(batch.sample_dose) if batch.sample_dose is not None else None
        return batch

    def save_sample_dose(self, sample_dose: NDArray[float], batch: DataBatch, epoch: int):
        sample_labels = self.oracle.label_dose_samples(sample_dose, batch)
        sample_dose_gy = self.historical_plan_bounds.get_dose_in_gray(sample_dose)
        all_samples_pass = sample_labels.all(axis=1)
        for patient_id, generated_patient_dose, is_pass in zip(batch.patient_list, sample_dose_gy, all_samples_pass):
            self.results_manager.sample_dose.write(generated_patient_dose, is_pass, patient_id, epoch)
