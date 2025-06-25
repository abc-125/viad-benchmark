import os
import numpy as np
import torch
from lightning.pytorch.loggers import MLFlowLogger

from anomalib import LearningType
from anomalib.models.components import AnomalyModule
from anomalib.engine import Engine

from .base import BaseExperiment



class ResultsReader(AnomalyModule):
    """Read anomaly maps from models."""
    def __init__(
        self,
        model_name=None,
        datasets_folder="datasets",
        anomaly_maps_folder="an_maps",
    ) -> None:
        super().__init__()
        self.model_name = model_name.lower()
        self.datasets_folder = datasets_folder
        self.anomaly_maps_folder = anomaly_maps_folder
            
    def configure_optimizers(self):
        # skip training
        return None

    def training_step(self, batch, batch_idx):
        # skip training
        return None

    def validation_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs):
        # get anomaly map paths
        img_paths = batch["image_path"]
        an_paths = [path.replace(
            self.datasets_folder, 
            os.path.join(self.anomaly_maps_folder, self.model_name)
        ) for path in img_paths]

        # read anomaly maps
        anomaly_maps = []
        for i, path in enumerate(an_paths):
            dir = os.path.dirname(path)
            an_path = os.path.join(dir, os.path.basename(path)[:-3] + "npy")
            anomaly_map = np.load(an_path)
            anomaly_map = torch.tensor(anomaly_map, dtype=torch.float32, device=self.device)
            anomaly_maps.append(anomaly_map)            
        
        # add results to batch for further processing by anomalib
        anomaly_maps = torch.stack(anomaly_maps)
        batch["anomaly_maps"] = anomaly_maps
        batch["pred_scores"] = torch.max(anomaly_maps.reshape(anomaly_maps.shape[0], -1), axis=1).values
        return batch    

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
            anomaly_maps_folder="an_maps"
        ):
        """Setup the experiment."""
        self.model_name = model_name
        self.datasets_folder = datasets_folder
        self.anomaly_maps_folder = anomaly_maps_folder

    def run_single_training(self, seed, model_name, dataset_name):
        """Run the experiment."""    
        classes = self.datasets_cathegories[dataset_name]
                    
        # iterate over classes
        for cls in classes:
            datamodule = self.dataset_factory(dataset_name=dataset_name, cls=cls)
            model = ResultsReader(
                model_name=model_name,
                datasets_folder=self.datasets_folder,
                anomaly_maps_folder=self.anomaly_maps_folder
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
            engine.validate(datamodule=datamodule, model=model)
            engine.test(datamodule=datamodule, model=model)