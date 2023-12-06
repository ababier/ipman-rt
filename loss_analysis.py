from matplotlib import pyplot as plt

from algorithms import Evaluation
from results_manager import ResultsManager

if __name__ == "__main__":
    """ "Rough script to visualize and analyze loss of models."""

    # Classifier
    classifier_results_manager = ResultsManager("baseline-final", model_name="classifier")
    df = classifier_results_manager.loss.read_all()
    means = df.groupby(["lambda_", "iteration", "epoch"], as_index=False, dropna=False).mean()
    df.binary_crossentropy.reset_index(drop=True).plot()
    plt.show()

    # IPMAN
    classifier_results_manager = ResultsManager("baseline-final", model_name="ipman")
    df_ipman = classifier_results_manager.loss.read_all()
    ipman_means = df.groupby(["lambda_", "iteration", "epoch"], as_index=False, dropna=False).mean()
    qs = df_ipman.groupby(["lambda_", "iteration", "epoch"], as_index=False, dropna=False).quantile(0.5)
