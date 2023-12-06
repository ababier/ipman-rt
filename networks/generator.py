from __future__ import annotations

from typing import Optional

import numpy as np
from keras import Model
from keras.layers import Activation, Add, AveragePooling3D, Conv3DTranspose, GaussianNoise, Input, Multiply, Rescaling, concatenate
from keras.optimizers.legacy import Adam
from open_kbp import DataBatch, DataShapes

from networks.base import ModelWrapper
from networks.network_blocks import make_block
from networks.reader_writer import ReaderWriter


class Generator(ModelWrapper):
    @classmethod
    def read(
        cls, data_shapes: DataShapes, experiment_id: str, name: str, lambda_: Optional[int] = None, iteration: Optional[int] = None, epoch: int = 0
    ) -> Generator:
        model = cls.define(data_shapes, name)
        if epoch:
            reader_writer = ReaderWriter(model, experiment_id, lambda_, iteration, epoch)
            reader_writer.read_model_weights()
            reader_writer.read_optimizer_weights()
        return cls(model)

    def write(self, experiment_id: str, lambda_: Optional[int] = None, iteration: Optional[int] = None, epoch: int = 0):
        reader_writer = ReaderWriter(self.model, experiment_id, lambda_, iteration, epoch)
        reader_writer.write_model_weights()
        reader_writer.write_optimizer_weights()

    @classmethod
    def define(
        cls,
        data_shapes: DataShapes,
        name: str = "generator",
        base_filter_num: int = 64,
        kernel_size: tuple[int, int, int] = (4, 4, 4),
        strides: tuple[int, int, int] = (2, 2, 2),
    ) -> Model:

        # Define inputs
        possible_dose_mask = Input(data_shapes.possible_dose_mask)
        ct_image = Input(data_shapes.ct)
        roi_masks = Input(data_shapes.structure_masks)

        # Build Model starting with Conv3D layers
        x0 = concatenate([ct_image, roi_masks])
        x1 = make_block(x0, base_filter_num, kernel_size, strides, relu_alpha=0.2)
        x2 = make_block(x1, 2 * base_filter_num, kernel_size, strides, relu_alpha=0.2)
        x3 = make_block(x2, 4 * base_filter_num, kernel_size, strides, relu_alpha=0.2)
        x4 = make_block(x3, 8 * base_filter_num, kernel_size, strides, relu_alpha=0.2)
        x5 = make_block(x4, 8 * base_filter_num, kernel_size, strides, relu_alpha=0.2)
        x6 = make_block(x5, 8 * base_filter_num, kernel_size, strides, relu_alpha=0.2, batch_norm=False)

        # Build model back up from bottleneck
        x5b = make_block(x6, 8 * base_filter_num, kernel_size, strides, relu_alpha=0.0, layer=Conv3DTranspose)
        x4b = make_block(x5b, 8 * base_filter_num, kernel_size, strides, dropout_rate=0.5, relu_alpha=0.0, x_skip=x5, layer=Conv3DTranspose)
        x3b = make_block(x4b, 4 * base_filter_num, kernel_size, strides, dropout_rate=0.5, relu_alpha=0.0, x_skip=x4, layer=Conv3DTranspose)
        x2b = make_block(x3b, 2 * base_filter_num, kernel_size, strides, relu_alpha=0.0, x_skip=x3, layer=Conv3DTranspose)
        x1b = make_block(x2b, base_filter_num, kernel_size, strides, relu_alpha=0.0, x_skip=x2, layer=Conv3DTranspose)
        x0b = make_block(x1b, 1, kernel_size, strides, batch_norm=False, x_skip=x1, layer=Conv3DTranspose)
        x0b_averaged = AveragePooling3D((3, 3, 3), strides=(1, 1, 1), padding="same")(x0b)

        # Apply mask to dose to zero areas that cannot physically receive dose (e.g., outside of patient).
        dose = Activation("tanh")(x0b_averaged)
        masked_dose = Multiply()([possible_dose_mask, dose])
        normalized_mask = Rescaling(scale=1, offset=-1)(possible_dose_mask)
        predicted_dose = Add()([masked_dose, normalized_mask])

        # Compile model for use
        model = Model(inputs=[possible_dose_mask, ct_image, roi_masks], outputs=predicted_dose, name=name)
        optimizer = Adam(learning_rate=2e-4, decay=1e-3, beta_1=0.5, beta_2=0.999)
        model.compile(loss="mean_absolute_error", optimizer=optimizer)
        return model

    def predict(self, batch: DataBatch):
        return np.array(self.model([batch.possible_dose_mask, batch.ct, batch.structure_masks]))

    def train(self, batch: DataBatch) -> float:
        ct_noise = GaussianNoise(stddev=0.05)(batch.ct)
        roi_noise = GaussianNoise(stddev=0.05)(batch.structure_masks)
        loss = self.model.train_on_batch([batch.possible_dose_mask, ct_noise, roi_noise], [batch.dose, batch.null_values])
        self.print_loss(loss)
        return loss
