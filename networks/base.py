from keras import Model
from keras import backend as keras_backend


class ModelWrapper:
    def __init__(self, model: Model):
        self.model = model

    @property
    def name(self) -> str:
        return self.model.name

    def print_loss(self, loss: float | list[float]):
        print(f"\ntraining loss for {self.model.name}:")
        if isinstance(loss, float):
            print(f"\t{self.model.loss} loss: {loss}")
        else:
            print("".join(f"\t{name} loss: {value:.3f}\n" for name, value in zip(self.loss_names, loss)))

    @property
    def loss_names(self) -> list[str]:
        return [self.model.loss] if isinstance(self.model.loss, str) else ["total", *self.model.loss]

    def reduce_iteration_count(self, factor: int = 2):
        opt_iteration = keras_backend.get_value(self.model.optimizer.iterations)
        keras_backend.set_value(self.model.optimizer.iterations, int(opt_iteration / factor))
