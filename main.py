from algorithms import CNNTrainer, Evaluation, GANTrainer, StageOne, StageTwo
from open_kbp import DataLoader
from oracle import Oracle
from results_manager import ResultsManager
from utils.historical_plan_bounds import HistoricalPlanBounds
from utils.main_arguments import MainArguments

if __name__ == "__main__":
    arguments = MainArguments()
    train_data_loader = DataLoader("train")
    historical_plan_bounds = HistoricalPlanBounds.get(train_data_loader, arguments.use_alternative_criteria)
    oracle = Oracle(historical_plan_bounds, arguments.use_alternative_criteria)
    results_manager = ResultsManager(arguments.experiment_id)
    train_data_loader.batch_size = 12
    lambdas = [256, 64, 16, 4]

    if arguments.train:
        evaluation = Evaluation(train_data_loader, oracle, results_manager, arguments.validate)

        # Train CNN baselines
        if arguments.train_cnn:
            cnn_trainer = CNNTrainer(train_data_loader, oracle, results_manager)
            cnn_trainer.run(epoch_limit=100)
            evaluation.run(model_name="cnn_generator", lambda_=None, iteration=None)

        # Train GAN baseline (required to initialize ipman generator and dose samples for iteration 1)
        gan_trainer = GANTrainer(train_data_loader, oracle, results_manager)
        gan_trainer.run(epoch_limit=100)
        evaluation.run(model_name="gan_generator", lambda_=None, iteration=None)

        # Train the ipman model for multiple iterations
        stage_one = StageOne(train_data_loader, oracle, results_manager)
        stage_two = StageTwo(train_data_loader, oracle, results_manager)
        for iteration in range(2, arguments.num_iterations):
            stage_one.run(iteration=iteration, epoch_limit=15)
            for lambda_ in lambdas:
                stage_two.run(lambda_=lambda_, l1_weight=arguments.l1_weight, iteration=iteration)
                evaluation.run(model_name="ipman_generator", lambda_=lambda_, iteration=iteration)

    # Evaluate model performance for both in-sample and out-of-sample data
    if arguments.test:
        evaluation = Evaluation(train_data_loader, oracle, results_manager, include_validation=False, include_test=arguments.test)
        evaluation.run(model_name="cnn_generator", lambda_=None, iteration=None)
        evaluation.run(model_name="gan_generator", lambda_=None, iteration=None)
        for iteration in range(1, arguments.num_iterations + 1):
            for lambda_ in lambdas:
                evaluation.run(model_name="ipman_generator", lambda_=lambda_, iteration=iteration)
