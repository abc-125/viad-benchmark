from experiments.eval import EvalExperiment


if __name__ == "__main__":
    experiment = EvalExperiment()
    experiment.setup(models=["Patchcore"], datasets=["BTech"])
    experiment.run()