from ctypes import c_void_p, c_wchar_p
from typing import Optional

from numpy.typing import NDArray
from open_kbp import DataBatch

from results_manager.evaluation import Evaluation
from results_manager.images import Images
from results_manager.loss import Loss
from results_manager.sample_dose import SampleDose


class ResultsManager:
    def __init__(self, experiment_id: str, lambda_: Optional[int] = None, iteration: Optional[int] = 0, model_name: Optional[str] = None):
        self.experiment_id = experiment_id
        self._model_name = c_wchar_p(model_name)
        self._lambda = c_void_p(lambda_)
        self._iteration = c_void_p(iteration)

        self.images = Images(self.experiment_id, self._model_name, self._lambda, self._iteration)
        self.loss = Loss(self.experiment_id, self._model_name, self._lambda, self._iteration)
        self.sample_dose = SampleDose(self.experiment_id, self._model_name, self._lambda, self._iteration)
        self.evaluation = Evaluation(self.experiment_id, self._model_name, self._lambda, self._iteration)

    @property
    def lambda_(self):
        return self._lambda.value

    @property
    def iteration(self):
        return self._iteration.value

    @property
    def model_name(self):
        return self._model_name.value

    def set_state(self, model_name: Optional[str], lambda_: Optional[int], iteration: Optional[int]):
        self._model_name.value = model_name
        self._lambda.value = lambda_
        self._iteration.value = iteration

    def log_loss(self, loss: float | list[float]) -> None:
        self.loss.log_loss(loss)

    def write_and_flush_loss_cache(self, epoch: int):
        self.loss.write_and_flush_cache(epoch)

    def write_images(self, batch: DataBatch, generated_dose: NDArray[float], epoch: int) -> None:
        self.images.make_dose_with_ct(generated_dose, batch, epoch)
