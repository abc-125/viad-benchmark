from anomalib.engine import Engine

from lightning.pytorch.loggers import MLFlowLogger

from .base import BaseExperiment


class EvalExperiment(BaseExperiment):
    """
    Class for evaluating models on a datasets (presented as datamodules).
    """
    def setup(self, models, datasets):
        """
        Setup the experiment.
        """
        self.models = models
        self.datasets = datasets
        self.experiment_name = "eval"
        self.seeds = [self.seeds[0]]  # Use only one seed for evaluation
        self.image_size = (256, 256)
        

    def run_single_training(self, seed, model_name, dataset_name):
        """
        Run the experiment.
        """
        classes = self.datasets_cathegories[dataset_name]
                    
        # iterate over classes
        for cls in classes:
            datamodule = self.dataset_factory(dataset_name=dataset_name, cls=cls)
            model = self.model_factory(model_name=model_name, image_size=self.image_size)
            run_name = f"{model_name}_{dataset_name}_{cls}_seed{seed}"
            engine = Engine(
                **self.default_params_trainer[model_name],
                # log results to mlflow
                logger=MLFlowLogger(
                    experiment_name=self.experiment_name,
                    run_name=run_name,
                    save_dir=self.mlflow_dir,
                    tags={
                            "seed": str(seed),
                            "dataset": dataset_name,
                            "cls": cls,
                            "model": model_name,
                        }
                    ),
                )
            engine.fit(datamodule=datamodule, model=model)
            engine.test(datamodule=datamodule, model=model)


