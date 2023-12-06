import argparse


class MainAnalysisArguments:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self._parse()
        if self.validate == 0 and self.test == 0:
            raise ValueError("At least one of the flags --validate or --test must be used to produce analysis")

    def _parse(self):
        self.parser.add_argument("-experiment_id", type=str, default="baseline", help="Used to identify experiment models and results.")
        self.parser.add_argument("--validate", action="store_true", help="Evaluate model performance on validation data.")
        self.parser.add_argument("--test", action="store_true", help="Evaluate model performance on test data.")
        self._args = self.parser.parse_args()

    @property
    def experiment_id(self) -> str:
        return self._args.experiment_id

    @property
    def validate(self) -> bool:
        return self._args.validate

    @property
    def test(self) -> bool:
        return self._args.test
