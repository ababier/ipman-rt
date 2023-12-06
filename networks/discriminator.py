import numpy as np
from keras import Model
from keras.layers import Activation, Input, Reshape, concatenate
from keras.optimizers.legacy import Adam
from numpy.typing import NDArray
from open_kbp import DataBatch, DataShapes

from networks.base import ModelWrapper
from networks.network_blocks import make_block
from networks.reader_writer import ReaderWriter


class Discriminator(ModelWrapper):
    @classmethod
    def read(cls, data_shapes: DataShapes, experiment_id: str, epoch: int = 0):
        model = cls.define(data_shapes)
        if epoch:
            reader_writer = ReaderWriter(model, experiment_id, epoch=epoch)
            reader_writer.read_model_weights()
            reader_writer.read_optimizer_weights()
        return cls(model)

    def write(self, experiment_id: str, epoch: int = 0):
        reader_writer = ReaderWriter(self.model, experiment_id, epoch=epoch)
        reader_writer.write_model_weights()
        reader_writer.write_optimizer_weights()

    @classmethod
    def define(
        cls,
        data_shapes: DataShapes,
        base_filter_num: int = 64,
        kernel_size: tuple[int, int, int] = (4, 4, 4),
        strides: tuple[int, int, int] = (2, 2, 2),
    ) -> Model:
        """Labels an input dose as reference (i.e., real) or generated (i.e., fake) based on a patient a ct image and roi masks"""
        # Define inputs
        dose_image = Input(data_shapes.dose)
        ct_image = Input(data_shapes.ct)
        roi_masks = Input(data_shapes.structure_masks)

        # Build Model
        x0 = concatenate([dose_image, ct_image, roi_masks])
        x1 = make_block(x0, base_filter_num, kernel_size, strides, relu_alpha=0.2)
        x2 = make_block(x1, 2 * base_filter_num, kernel_size, strides, relu_alpha=0.2, dropout_rate=0.5)
        x3 = make_block(x2, 4 * base_filter_num, kernel_size, strides, relu_alpha=0.2)
        x4 = make_block(x3, 8 * base_filter_num, kernel_size, strides, relu_alpha=0.2)
        x5 = make_block(x4, 16 * base_filter_num, kernel_size, strides, relu_alpha=0.2)
        x6 = make_block(x5, num_filters=1, kernel_size=kernel_size, strides=kernel_size, batch_norm=False)

        # Reshape model output to form probability prediction
        is_real_probability = Activation("sigmoid")(Reshape((1,))(x6))

        model = Model(inputs=[dose_image, ct_image, roi_masks], outputs=is_real_probability, name="discriminator")
        optimizer = Adam(learning_rate=2e-4, decay=1e-3, beta_1=0.5, beta_2=0.999)
        model.compile(loss="binary_crossentropy", optimizer=optimizer)
        return model

    def train(self, batch: DataBatch, fake_samples: NDArray) -> None:
        all_doses = np.concatenate((batch.dose, fake_samples))
        all_labels = np.concatenate((np.ones(batch.size, dtype=int), np.zeros(batch.size, dtype=int)))
        all_cts = np.repeat(batch.ct, 2, axis=0)
        all_roi_masks = np.repeat(batch.structure_masks, 2, axis=0)
        self.model.train_on_batch([all_doses, all_cts, all_roi_masks], [all_labels])
