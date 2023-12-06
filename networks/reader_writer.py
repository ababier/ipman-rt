import os
import pickle
from pathlib import Path
from typing import Optional

import tensorflow as tf
from keras import Model
from utils import EXPERIMENTS_DIR


class ReaderWriter:
    def __init__(self, model: Model, experiment_id: str, lambda_: Optional[int] = None, iteration: Optional[int] = None, epoch: int = 0):
        self.model = model
        self.experiment_id = experiment_id
        self.lambda_ = lambda_
        self.iteration = iteration
        self.epoch = self.get_last_epoch() if epoch == -1 else epoch

    @property
    def parent_path(self) -> Path:
        if self.lambda_ is None and self.iteration is None:
            return EXPERIMENTS_DIR / self.experiment_id / self.model.name
        if self.lambda_ is None:
            return EXPERIMENTS_DIR / self.experiment_id / self.model.name / f"iteration_{self.iteration}"
        return EXPERIMENTS_DIR / self.experiment_id / self.model.name / f"lambda_{self.lambda_}" / f"iteration_{self.iteration}"

    @property
    def model_weights_path(self) -> Path:
        return self.parent_path / f"epoch_{self.epoch}_model_weights.h5"

    @property
    def optimizer_weights_path(self) -> Path:
        return self.parent_path / f"epoch_{self.epoch}_optimizer_weights.h5"

    def write_model_weights(self) -> None:
        os.makedirs(self.model_weights_path.parent, exist_ok=True)
        self.model.save_weights(self.model_weights_path.as_posix())

    def write_optimizer_weights(self) -> None:
        os.makedirs(self.optimizer_weights_path.parent, exist_ok=True)
        with open(self.optimizer_weights_path, "wb") as file:
            pickle.dump(self.model.optimizer.get_weights(), file)

    def read_model_weights(self) -> None:
        if not self.model_weights_path.exists():
            raise ValueError(f"Model weights for {self.model.name} at epoch {self.epoch} are not saved at {self.model_weights_path}.")
        self.model.load_weights(self.model_weights_path)
        print(f"Read {self.model.name} model weights from {self.model_weights_path}.")

    def read_optimizer_weights(self) -> None:
        if not self.optimizer_weights_path.exists():
            raise ValueError(f"Optimizer weights for {self.model.name} at epoch {self.epoch} are not saved at {self.optimizer_weights_path}.")
        file_to_load = open(r"" + self.optimizer_weights_path.as_posix(), "rb")
        optimizer_weights = pickle.load(file_to_load)
        if len(optimizer_weights) > 0:
            self._initialize_optimizer_weights()
            self.model.optimizer.set_weights(optimizer_weights)
            print(f"Read {self.model.name} optimizer weights from {self.optimizer_weights_path}.")
        else:
            print(f"The optimizer weights are empty for {self.model.name} at {self.optimizer_weights_path}, using random initialization.")

    def _initialize_optimizer_weights(self) -> None:
        grad_vars = self.model.trainable_weights
        zero_grads = [tf.zeros_like(w) for w in grad_vars]
        self.model.optimizer.apply_gradients(list(zip(zero_grads, grad_vars)))

    def get_last_epoch(self) -> int:
        saved_epochs = (int(path.stem.split("_")[1]) for path in self.parent_path.iterdir())
        last_epoch = max(saved_epochs, default=0)
        return last_epoch
