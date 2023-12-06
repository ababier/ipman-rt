from networks import GAN

from algorithms.base import Algorithm


class GANTrainer(Algorithm):
    """Trains a GAN to produce a generator that we can refine with IPMAN, samples are also generated to train the classifier in Stage One."""

    def _run_iteration(self, epoch_limit: int = 100):
        self.data_loader.set_files_to_load("primary")

        gan = GAN.read(self.data_loader.data_shapes, self.experiment_id)
        self.results_manager.set_state(gan.name, lambda_=None, iteration=None)
        self.results_manager.loss.set_new_model(gan)

        for epoch in range(1, epoch_limit + 1):
            print(f"\nTraining {gan.model.name}, starting_epoch {epoch} of {epoch_limit}")
            for batch in self.data_loader.get_batches():
                batch = self.normalize_batch_dose(batch)
                generated_dose = gan.generator.predict(batch)
                gan.discriminator.train(batch, generated_dose)
                loss = gan.train(batch)
                self.results_manager.log_loss(loss)
                self.results_manager.write_images(batch, generated_dose, epoch)
                self.save_sample_dose(generated_dose, batch, epoch)
            self.results_manager.write_and_flush_loss_cache(epoch)

        gan.write(self.experiment_id, epoch_limit)
