import numpy as np
from keras import Model
from keras.layers import Activation, Dense, Flatten, Input, MaxPool3D, Reshape, concatenate
from keras.optimizers.legacy import Adam
from numpy.typing import NDArray
from open_kbp import DataBatch, DataShapes

from networks.base import ModelWrapper
from networks.network_blocks import make_block
from networks.reader_writer import ReaderWriter


class Classifier(ModelWrapper):
    @classmethod
    def read(cls, data_shapes: DataShapes, experiment_id: str, iteration: int = 0, epoch: int = -1):
        model = cls.define(data_shapes)
        if iteration > 1:
            reader_writer = ReaderWriter(model, experiment_id, iteration=iteration, epoch=epoch)
            reader_writer.read_model_weights()
            reader_writer.read_optimizer_weights()
        return cls(model)

    def write(self, experiment_id: str, iteration: int = 0, epoch: int = 0):
        reader_writer = ReaderWriter(self.model, experiment_id, iteration=iteration, epoch=epoch)
        reader_writer.write_model_weights()
        reader_writer.write_optimizer_weights()

    @classmethod
    def define(
        cls,
        data_shapes: DataShapes,
        base_filter_num: int = 32,
        kernel_size: tuple[int, int, int] = (4, 4, 4),
        strides: tuple[int, int, int] = (2, 2, 2),
    ) -> Model:
        """Makes a classifier that predicts whether a dose distribution satisfies all appropriate constraints for each ROI"""
        # Define inputs
        dose_image = Input(data_shapes.dose)
        ct_image = Input(data_shapes.ct)
        roi_masks = Input(data_shapes.structure_masks)

        # Convolution layers
        x0 = concatenate([dose_image, ct_image, roi_masks])
        x1 = make_block(x0, base_filter_num, kernel_size, strides, relu_alpha=0.2)
        x2 = make_block(x1, 2 * base_filter_num, kernel_size, strides, relu_alpha=0.2)
        x3 = make_block(x2, 4 * base_filter_num, kernel_size, strides=(1, 1, 1), relu_alpha=0.2)
        x4 = make_block(x3, strides=strides, relu_alpha=0.2, layer=MaxPool3D, x_skip=x2)
        x5 = make_block(x4, 8 * base_filter_num, kernel_size, strides, relu_alpha=0.2)
        x6 = make_block(x5, 16 * base_filter_num, kernel_size, strides=(1, 1, 1), relu_alpha=0.2)
        x7 = make_block(x6, strides=strides, relu_alpha=0.2, layer=MaxPool3D, x_skip=x5)
        x8 = make_block(x7, 16 * base_filter_num, kernel_size, kernel_size, relu_alpha=0.2)

        # Flatten final convolution output to form roi-specific probabilities
        x9 = Dense(data_shapes.num_rois)(Flatten()(x8))
        x10 = Reshape((data_shapes.num_rois, 1))(x9)
        constraint_satisfied_probability = Activation("sigmoid")(x10)

        # Assemble model
        model = Model(inputs=[dose_image, ct_image, roi_masks], outputs=constraint_satisfied_probability, name="classifier")
        optimizer = Adam(learning_rate=2e-4, decay=1e-3, beta_1=0.5, beta_2=0.999)
        model.compile(loss="binary_crossentropy", optimizer=optimizer, sample_weight_mode="temporal")
        return model

    def train(self, batch: DataBatch, labels: NDArray[float]) -> float:
        non_nan_labels = np.nan_to_num(labels, nan=1).astype(int)
        sample_weights = self._get_sample_weights(non_nan_labels)
        loss = self.model.train_on_batch([batch.dose, batch.ct, batch.structure_masks], non_nan_labels, sample_weight=sample_weights)
        self.print_loss(loss)
        return loss

    @staticmethod
    def _get_sample_weights(all_labels: NDArray[int]) -> NDArray[float]:
        all_sample_weights = []
        for labels in all_labels.squeeze(axis=-1).T:
            label_counts = np.bincount(labels)
            labels_by_counts = label_counts[labels]
            sample_weights = max(labels_by_counts) / labels_by_counts
            all_sample_weights.append(sample_weights)
        return np.array(all_sample_weights).T

    def predict(self, batch: DataBatch):
        return self.model.predict_on_batch([batch.dose, batch.ct, batch.structure_masks])
