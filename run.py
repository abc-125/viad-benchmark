from src.experiments.eval import EvalExperiment


if __name__ == "__main__":
    # constants for all experiments
    models = ["Patchcore", "Fastflow", "ReverseDistillation"]
    datasets = ["BTech", "VAD", "Visa", "MPDD"]

    # create an experiment
    experiment = EvalExperiment()
    experiment_name = "eval"

    # setup experiment with selected models and datasets, pass metrics
    experiment.setup(
        models=models, 
        datasets=datasets,
        experiment_name=experiment_name,
    )

    # run training and evaluation
    experiment.run()