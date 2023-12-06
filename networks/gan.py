from keras import Model
from keras.layers import GaussianNoise, Input
from keras.optimizers.legacy import Adam
from open_kbp import DataBatch, DataShapes

from networks.base import ModelWrapper
from networks.discriminator import Discriminator
from networks.generator import Generator
from networks.loss_functions import log_predicted_value_loss
from networks.reader_writer import ReaderWriter


class GAN(ModelWrapper):
    def __init__(self, model: Model, generator: Generator, discriminator: Discriminator):
        super().__init__(model)
        self.generator = generator
        self.discriminator = discriminator

    @classmethod
    def read(cls, data_shapes: DataShapes, experiment_id: str, epoch: int = 0):
        generator = Generator.read(data_shapes, experiment_id, name="gan_generator", epoch=epoch)
        discriminator = Discriminator.read(data_shapes, experiment_id, epoch=epoch)
        model = cls.define(data_shapes, generator, discriminator)
        if epoch:
            discriminator.model.trainable = False
            reader_writer = ReaderWriter(model, experiment_id, epoch=epoch)
            reader_writer.read_optimizer_weights()
            discriminator.model.trainable = True
        return cls(model, generator, discriminator)

    def write(self, experiment_id: str, epoch: int):
        self.generator.write(experiment_id, epoch=epoch)
        self.discriminator.write(experiment_id, epoch=epoch)
        reader_writer = ReaderWriter(self.model, experiment_id, epoch=epoch)
        self.discriminator.model.trainable = False
        reader_writer.write_optimizer_weights()
        self.discriminator.model.trainable = True

    @classmethod
    def define(cls, data_shapes: DataShapes, generator: Generator, discriminator: Discriminator) -> Model:
        # Define inputs
        possible_dose_mask = Input(data_shapes.possible_dose_mask)
        ct_image = Input(data_shapes.ct)
        roi_masks = Input(data_shapes.structure_masks)

        # Model operations
        ct_noise = GaussianNoise(stddev=0.05)(ct_image)
        roi_noise = GaussianNoise(stddev=0.05)(roi_masks)
        generated_dose = generator.model([possible_dose_mask, ct_noise, roi_noise])
        generated_dose_label = discriminator.model([generated_dose, ct_image, roi_masks])

        # Assemble model
        model = Model(inputs=[possible_dose_mask, ct_image, roi_masks], outputs=[generated_dose, generated_dose_label], name="gan")
        losses = {generator.model.name: "mean_absolute_error", discriminator.model.name: log_predicted_value_loss}
        loss_weights = {generator.model.name: 90, discriminator.model.name: -1}
        discriminator.model.trainable = False
        model.compile(loss=losses, loss_weights=loss_weights, optimizer=Adam(learning_rate=2e-4, decay=1e-3, beta_1=0.5, beta_2=0.999))
        discriminator.model.trainable = True
        return model

    def train(self, batch: DataBatch) -> float:
        self.discriminator.trainable = False
        loss = self.model.train_on_batch([batch.possible_dose_mask, batch.ct, batch.structure_masks], [batch.dose, batch.null_values])
        self.discriminator.trainable = True
        self.print_loss(loss)
        return loss
