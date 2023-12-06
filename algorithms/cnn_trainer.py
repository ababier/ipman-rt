from networks import Generator

from algorithms.base import Algorithm


class CNNTrainer(Algorithm):
    """Trains a CNN to produce a generator that we can refine with IPMAN."""

    def _run_iteration(self, epoch_limit: int = 100):
        self.data_loader.set_files_to_load("primary")

        generator = Generator.read(self.data_loader.data_shapes, self.experiment_id, name="cnn_generator")
        self.results_manager.set_state(model_name=generator.name, lambda_=None, iteration=None)
        self.results_manager.loss.set_new_model(generator)

        for epoch in range(1, epoch_limit + 1):
            print(f"\nTraining CNN, starting_epoch {epoch} of {epoch_limit}")
            for batch in self.data_loader.get_batches():
                batch = self.normalize_batch_dose(batch)
                loss = generator.train(batch)
                self.results_manager.log_loss(loss)
            self.results_manager.write_and_flush_loss_cache(epoch)

        generator.write(self.experiment_id, epoch=epoch_limit)
