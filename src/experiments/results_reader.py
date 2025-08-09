import os
import numpy as np
import torch
from lightning.pytorch.loggers import MLFlowLogger
import pandas as pd

from anomalib import LearningType
from anomalib.models.components import AnomalibModule
from anomalib.engine import Engine
from anomalib.data.dataclasses.torch import InferenceBatch, ImageBatch
from anomalib.metrics import Evaluator

from .base import BaseExperiment


class ResultsReader(AnomalibModule):
    """Read anomaly maps from models."""

    def __init__(
            self,
            model_name=None,
            datasets_folder="datasets",
            anomaly_maps_folder="an_maps",
            experiment_name="",
            evaluator=None,
    ) -> None:
        super().__init__()
        self.model_name = model_name.lower()
        self.datasets_folder = datasets_folder
        self.anomaly_maps_folder = anomaly_maps_folder
        self.experiment_name = experiment_name
        self.evaluator = evaluator

    def configure_optimizers(self):
        # skip training
        return None

    def training_step(self, batch, batch_idx):
        # skip training
        return None

    def validation_step(self, batch, *args, **kwargs):
        # get anomaly map paths
        img_paths = batch.image_path
        an_paths = [path.replace(
            self.datasets_folder,
            os.path.join(self.anomaly_maps_folder, self.model_name + "_" + self.experiment_name)
        ) for path in img_paths]

        # paths in csv - use different root format
        an_paths_csv = [path.replace(
            self.datasets_folder,
            os.path.join(self.anomaly_maps_folder, self.model_name)
        ).replace("/srv/data/Work/viad-benchmark/", "../") for path in img_paths]

        # csv
        csv_path = "/".join(an_paths[0].split('/')[:-3]) + "/results.csv"
        df = pd.read_csv(csv_path)

        anomaly_maps = []
        scores = []
        for i, path in enumerate(an_paths):
            # get anomaly map
            an_path = path[:-3] + "npy"
            anomaly_map = np.load(an_path)
            anomaly_map = torch.tensor(anomaly_map, dtype=torch.float32, device=self.device)
            anomaly_maps.append(anomaly_map)

            # get score
            score = df.loc[df['path'] == an_paths_csv[i], 'scores'].iloc[0]
            scores.append(torch.tensor(score, dtype=torch.float32, device=self.device))

        # add results to batch for further processing by anomalib
        predictions = InferenceBatch(
            anomaly_map=torch.stack(anomaly_maps),
            pred_score=torch.stack(scores),
        )
        return batch.update(**predictions._asdict())

    @property
    def trainer_arguments(self):
        return {"num_sanity_val_steps": 0, "max_epochs": 1, "check_val_every_n_epoch": 1}

    @property
    def learning_type(self):
        """Return the learning type of the model.

        Returns:
            LearningType: Learning type of the model.
        """
        return LearningType.ONE_CLASS


class ResultsReaderExperiment(BaseExperiment):
    """Experiment for reading anomaly maps from models."""

    def setup(
            self,
            model_name,
            datasets_folder="datasets",
            anomaly_maps_folder="an_maps",
            experiment_name="",
            metrics=None,
            models=[],
            datasets=[],
    ):
        """Setup the experiment."""
        self.model_name = model_name
        self.datasets_folder = datasets_folder
        self.anomaly_maps_folder = anomaly_maps_folder
        self.experiment_name = experiment_name
        self.seeds = [self.seeds[0]]
        self.metrics = metrics
        self.models = models
        self.datasets = datasets

    def run_single_training(self, seed, model_name, dataset_name):
        """Run the experiment."""
        classes = self.datasets_cathegories[dataset_name]

        # iterate over classes
        for cls in classes:
            datamodule = self.dataset_factory(dataset_name=dataset_name, cls=cls)
            evaluator = Evaluator(test_metrics=self.metrics, compute_on_cpu=False)
            model = ResultsReader(
                model_name=model_name,
                datasets_folder=self.datasets_folder,
                anomaly_maps_folder=self.anomaly_maps_folder,
                experiment_name=self.experiment_name,
                evaluator=evaluator,
            )
            run_name = f"{model_name}_{dataset_name}_{cls}_seed{seed}"
            default_root_dir = os.path.join(
                self.root,
                "results",
                "eval_" + self.experiment_name,
                model_name,
                dataset_name,
                cls,
                "v0"
            )
            engine = Engine(
                default_root_dir=default_root_dir,
                # log results to mlflow
                logger=MLFlowLogger(
                    experiment_name="eval_" + self.experiment_name,
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
            engine.validate(datamodule=datamodule, model=model)
            engine.test(datamodule=datamodule, model=model)
            