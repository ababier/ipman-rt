from networks import Classifier

from algorithms.base import Algorithm


class StageOne(Algorithm):
    """Stage one is the classifier training stage, which trains a classifier to predict if constraints for each ROI are satisfied by a given dose."""

    def _run_iteration(self, iteration: int, epoch_limit: int = 10) -> None:
        classifier = Classifier.read(self.data_loader.data_shapes, self.experiment_id, iteration - 1)
        classifier.reduce_iteration_count()
        self.results_manager.set_state(classifier.name, lambda_=None, iteration=iteration)
        self.results_manager.loss.set_new_model(classifier)
        self.setup_data_loader()
        self.data_loader.batch_size = min(4 * self.data_loader.batch_size, 32)

        for epoch in range(1, epoch_limit + 1):
            print(f"Iteration {iteration}, starting_epoch {epoch} of {epoch_limit}.")
            for batch in self.data_loader.get_batches():
                batch = self.normalize_batch_dose(batch)
                labels = self.oracle.label_dose_samples(batch.sample_dose, batch)
                loss = classifier.train(batch, labels)
                self.results_manager.log_loss(loss)
            self.results_manager.write_and_flush_loss_cache(epoch)
        classifier.write(self.experiment_id, iteration, epoch_limit)

    def setup_data_loader(self) -> None:
        self.data_loader.set_files_to_load("with_sample")
        sample_paths = self.results_manager.sample_dose.get_recent_sample_dose_paths()
        self.data_loader.dose_sample_paths = sample_paths
        print(f"Added {len(sample_paths)} samples to data loader")
