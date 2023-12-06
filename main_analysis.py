import os

from analysis import plot_all_from_summary_csv, plot_criteria_satisfaction
from results_manager import ResultsManager
from utils import EXPERIMENTS_DIR
from utils.main_analysis_arguments import MainAnalysisArguments

if __name__ == "__main__":
    arguments = MainAnalysisArguments()
    plots_directory = EXPERIMENTS_DIR / arguments.experiment_id / "analysis"
    results_manager = ResultsManager(experiment_id=arguments.experiment_id)
    summary = results_manager.evaluation.read_all()
    summary.filter_by_dataset(validation=arguments.validate, test=arguments.test)

    # Model criteria (last epoch and last iteration)
    model_summaries = summary.summarize_final_iterations()
    os.makedirs(plots_directory, exist_ok=True)
    model_summaries.to_csv(plots_directory / "model_summaries.csv")

    # Plot iteration progress
    summary.set_iteration_one_for_lambdas()
    plot_all_from_summary_csv(summary, plots_directory)
    plot_criteria_satisfaction(summary, plots_directory)
