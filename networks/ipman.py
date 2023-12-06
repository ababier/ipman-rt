from typing import Optional

import tensorflow as tf
from keras import Model
from keras.layers import Input, Lambda
from keras.optimizers.legacy import Adam
from open_kbp import DataBatch, DataShapes

from networks.base import ModelWrapper
from networks.classifier import Classifier
from networks.generator import Generator
from networks.loss_functions import calculate_ipm_objective, minimize_value, sum_log_predicted_value_loss
from networks.reader_writer import ReaderWriter


class IPMAN(ModelWrapper):
    def __init__(self, model: Model, generator: Generator, classifier: Classifier):
        super().__init__(model)
        self.generator = generator
        self.classifier = classifier

    @classmethod
    def read(cls, data_shapes: DataShapes, experiment_id: str, lambda_: int, iteration: int, l1_weight: int):
        generator = cls._get_last_generator(data_shapes, experiment_id, lambda_, iteration)
        classifier = Classifier.read(data_shapes, experiment_id, iteration + 1, epoch=-1)  # classifier is pre-trained for current iteration
        model = cls.define(generator, classifier, data_shapes, lambda_, l1_weight)
        if iteration > 1:
            classifier.model.trainable = False
            reader_writer = ReaderWriter(model, experiment_id, lambda_, iteration, epoch=-1)
            reader_writer.read_optimizer_weights()
            classifier.model.trainable = True
        return cls(model, generator, classifier)

    @staticmethod
    def _get_last_generator(data_shapes: DataShapes, experiment_id: str, lambda_: Optional[int], previous_iteration: int) -> Generator:
        generator_to_load = "ipman_generator" if previous_iteration > 1 else "gan_generator"
        lambda_to_load = lambda_ if previous_iteration > 1 else None
        iteration_to_load = previous_iteration if previous_iteration > 1 else None
        generator = Generator.read(data_shapes, experiment_id, generator_to_load, lambda_to_load, iteration_to_load, epoch=-1)
        generator.model._name = "ipman_generator"
        return generator

    def write(self, experiment_id: str, lambda_: float, iteration: int, epoch: int):
        self.generator.write(experiment_id, lambda_, iteration, epoch)
        reader_writer = ReaderWriter(self.model, experiment_id, lambda_, iteration, epoch)
        self.classifier.model.trainable = False
        reader_writer.write_optimizer_weights()
        self.classifier.model.trainable = True

    @classmethod
    def define(cls, generator: Generator, classifier: Classifier, data_shapes: DataShapes, lambda_weight: int, l1_weight: int) -> Model:
        # Define inputs
        possible_dose_mask = Input(data_shapes.possible_dose_mask)
        ct_image = Input(data_shapes.ct)
        roi_masks = Input(data_shapes.structure_masks)

        # Predict dose with generator
        generated_dose = generator.model([possible_dose_mask, ct_image, roi_masks])
        objective_value = Lambda(
            lambda x: tf.map_fn(lambda y: calculate_ipm_objective(*y, data_shapes), x, fn_output_signature=tf.float32), name="objective"
        )([generated_dose, roi_masks])
        dose_labels = classifier.model([generated_dose, ct_image, roi_masks])

        # Assemble model
        classifier.model.trainable = False
        model = Model(inputs=[possible_dose_mask, ct_image, roi_masks], outputs=[objective_value, generated_dose, dose_labels], name="ipman")
        losses = {
            "objective": minimize_value,
            generator.model.name: "mean_absolute_error",
            classifier.model.name: sum_log_predicted_value_loss,
        }
        loss_weights = {"objective": 1 / lambda_weight, generator.model.name: l1_weight, classifier.model.name: -1}
        model.compile(loss=losses, loss_weights=loss_weights, optimizer=Adam(learning_rate=2e-6, decay=1e-3, beta_1=0.5, beta_2=0.999))
        classifier.model.trainable = True
        return model

    def train(self, batch: DataBatch) -> list[float]:
        self.classifier.model.trainable = False
        loss = self.model.train_on_batch(
            [batch.possible_dose_mask, batch.ct, batch.structure_masks], [batch.null_values, batch.dose, batch.null_values]
        )
        self.classifier.model.trainable = True
        self.print_loss(loss)
        return loss
