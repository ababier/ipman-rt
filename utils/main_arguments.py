import argparse


class MainArguments:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self._parse()

    def _parse(self):
        self.parser.add_argument("-experiment_id", type=str, default="baseline", help="Used to identify experiment models and results.")
        self.parser.add_argument("-num_iterations", type=int, default=11, help="Number of iterations that IPMAN is trained for.")
        self.parser.add_argument("-l1_weight", type=int, default=50, help="The weight assigned to the l1 term in IPMAN loss function.")
        self.parser.add_argument("--use_alternative_criteria", action="store_true", help="Make oracle use alternative criteria.")
        self.parser.add_argument("--train", action="store_true", help="Train models with training data.")
        self.parser.add_argument("--validate", action="store_true", help="Evaluate model performance on validation data.")
        self.parser.add_argument("--test", action="store_true", help="Evaluate model performance on test data.")
        self.parser.add_argument("--train_cnn", action="store_true", help="Train CNN baseline model.")
        self._args = self.parser.parse_args()

    @property
    def experiment_id(self) -> str:
        return self._args.experiment_id

    @property
    def num_iterations(self) -> int:
        return self._args.num_iterations

    @property
    def use_alternative_criteria(self) -> bool:
        return self._args.use_alternative_criteria

    @property
    def l1_weight(self) -> int:
        return self._args.l1_weight

    @property
    def train(self) -> bool:
        return self._args.train

    @property
    def validate(self) -> bool:
        return self._args.validate

    @property
    def test(self) -> bool:
        return self._args.test

    @property
    def train_cnn(self) -> bool:
        return self._args.train_cnn
