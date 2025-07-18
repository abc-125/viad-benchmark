from abc import ABC, abstractmethod
import os
from lightning.pytorch import seed_everything

from anomalib.metrics import AUROC, AUPRO, F1Score, Evaluator
from anomalib.data import BTech, Visa, VAD, MPDD
from anomalib.models import Patchcore, ReverseDistillation, Csflow, Fastflow
from anomalib.pre_processing import PreProcessor



class BaseExperiment(ABC):
    """
    Base class for experiments.
    """

    def __init__(self, root="./"):
        # seeds
        self.seeds = [3341, 1954, 1087]

        # root directory
        self.root = root

        # logger directory
        self.mlflow_dir = os.path.join(self.root, "mlruns")

        #metrics
        self.metrics = [
            AUROC(fields=["pred_score", "gt_label"], prefix="image_"),
            F1Score(fields=["pred_score", "gt_label"], prefix="image_"),
            AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),
            AUPRO(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),
            F1Score(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),
        ]

        # categories for each dataset
        self.datasets_cathegories = {
            "BTech": ["01", "02", "03"],
            "VAD": ["vad"],
            "Visa": ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", 
                     "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"],
            "MPDD": ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]
        }

        # model parameters for training based on papers
        self.default_params_trainer = {
            "Patchcore": {},
            "ReverseDistillation": {"max_epochs": 200, "check_val_every_n_epoch": 200},
            "Csflow": {"max_epochs": 240, "check_val_every_n_epoch": 240},
            "Fastflow": {"max_epochs": 200, "check_val_every_n_epoch": 200},
        }

        # base experiment parameters
        self.models = []
        self.datasets = []
        self.experiment_name = ""


    def dataset_factory(self, dataset_name, cls):
        """
        Factory method to create a dataset instance based on the dataset name and class.
        """
        classes = {
            "BTech": BTech,
            "VAD": VAD,
            "Visa": Visa,
            "MPDD": MPDD,
        }
        dataset_class = classes.get(dataset_name)
        if dataset_class is None:
            raise ValueError(
                f"Dataset '{dataset_name}' is not recognized. Available datasets are: {list(classes.keys())}"
            )
        
        dataset = dataset_class(
            root=os.path.join(self.root, "datasets", dataset_name.lower()), 
            category=cls, 
            train_batch_size=16
        )
        
        return dataset
    

    def model_factory(self, model_name, image_size, transform=None):
        """
        Factory method to create a model instance based on the model name.
        The model is created with the pre-processor configured for the given image size
        and the evaluator configured with the specified metrics.
        """
        classes = {
            "Patchcore": Patchcore,
            "ReverseDistillation": ReverseDistillation,
            "Csflow": Csflow,
            "Fastflow": Fastflow,
        }
        model_class = classes.get(model_name)
        if model_class is None:
            raise ValueError(
                f"Model '{model_name}' is not recognized. Available models are: {list(classes.keys())}"
            )
        
        if transform is None:
            pre_processor = model_class.configure_pre_processor(image_size=image_size)
        else:
            pre_processor = PreProcessor(transform=transform)
            print(f"Using custom transform.")

        evaluator = Evaluator(test_metrics=self.metrics, compute_on_cpu=False)

        return model_class(evaluator=evaluator, pre_processor=pre_processor)


    def run(self):
        """
        Method to run the training over multiple (seed, model, dataset).
        """
        for seed in self.seeds:
            seed_everything(seed, workers=True)
            for model in self.models:
                for dataset in self.datasets:
                    self.run_single_training(
                        seed=seed, model_name=model, dataset_name=dataset
                    )


    @abstractmethod
    def run_single_training(self, seed, model_name, dataset_name):
        """
        Run the experiment.
        """
        raise NotImplementedError
