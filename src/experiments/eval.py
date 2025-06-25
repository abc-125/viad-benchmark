import os
from lightning.pytorch.loggers import MLFlowLogger

from anomalib.engine import Engine

from .base import BaseExperiment


class EvalExperiment(BaseExperiment):
    """Evaluating models on a datasets."""
    def setup(
            self, 
            models, 
            datasets, 
            experiment_name="eval", 
            image_size=(256, 256),
            transform=None, 
            metrics=None
        ):
        """Setup the experiment."""
        # custom parameters
        self.models = models
        self.datasets = datasets
        self.experiment_name = experiment_name
        self.transform = transform
        self.image_size = image_size
        if metrics is not None:
            self.metrics = metrics
        
        # default parameters
        self.seeds = [self.seeds[0]]  # use only one seed for evaluation
        
    def run_single_training(self, seed, model_name, dataset_name):
        """Run the single training of the experiment."""
        classes = self.datasets_cathegories[dataset_name]
                    
        # iterate over classes
        for cls in classes:
            datamodule = self.dataset_factory(dataset_name=dataset_name, cls=cls)
            model = self.model_factory(
                model_name=model_name, 
                image_size=self.image_size, 
                transform=self.transform
            )
            run_name = f"{model_name}_{dataset_name}_{cls}_seed{seed}"
            engine = Engine(
                **self.default_params_trainer[model_name],
                default_root_dir=os.path.join(self.root, "results", self.experiment_name),
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


