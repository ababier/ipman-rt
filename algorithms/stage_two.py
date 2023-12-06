from networks import IPMAN

from algorithms.base import Algorithm


class StageTwo(Algorithm):
    """Stage two is where we train the generator with help from the classifier trained in stage one.
    The generator is trained to produce dose distributions that satisfy constraints based on a barrier created from the classifier."""

    def _run_iteration(self, lambda_: int, l1_weight: int, iteration: int, epoch_limit: int = 1):
        """
        Args:
            lambda_: 1/lambda_ is the weight placed on the barrier part of the IPMAN loss function.
            l1_weight: the weight placed on the l1 part of the IPMAN loss function.
            iteration: the iteration that was trained.
            epoch_limit: the number of epochs that the current iteration will be trained for.
        """
        ipman = IPMAN.read(self.data_loader.data_shapes, self.experiment_id, lambda_, iteration - 1, l1_weight)
        self.results_manager.set_state(ipman.name, lambda_, iteration)
        self.data_loader.set_files_to_load("primary")
        self.results_manager.loss.set_new_model(ipman)

        for epoch in range(1, epoch_limit + 1):
            print(f"Iteration {iteration} for lambda {lambda_}: epoch {epoch} of {epoch_limit} of stage two.")
            for batch in self.data_loader.get_batches():
                batch = self.normalize_batch_dose(batch)
                loss = ipman.train(batch)
                self.results_manager.log_loss(loss)
                generated_dose = ipman.generator.predict(batch)
                self.results_manager.write_images(batch, generated_dose, epoch)
                self.save_sample_dose(generated_dose, batch, epoch)
            self.results_manager.write_and_flush_loss_cache(epoch)
        ipman.write(self.experiment_id, lambda_, iteration, epoch_limit)
