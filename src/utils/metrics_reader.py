import os
import pandas as pd
import yaml


class MetricsReader():
    """
    Class for reading MLFlow metrics.
    """
    def __init__(self, mlruns_path="./mlruns"):
        self.mlruns_path = mlruns_path
        self.runs = {}

    def get_experiment_name(self, experiment_path: str) -> str:
        """
        Get the experiment name from the meta.yaml file in the experiment folder.
        If the file does not exist, return None.
        """
        meta_file = os.path.join(experiment_path, "meta.yaml")
        if os.path.exists(meta_file):
            with open(meta_file, 'r') as f:
                meta_data = yaml.safe_load(f)
                return meta_data.get("name", None)
        return None


    def iterate_experiments(self) -> dict:
        """
        Iterate over the experiments in the mlruns folder
        and return a dictionary of all experiment names and paths.
        """

        experiments = {}
        experiments_folders = os.listdir(self.mlruns_path)

        # remove mlflow system folders
        experiments_folders = [folder for folder in experiments_folders if folder not in [".trash", "0"]]

        # iterate over experiment folders
        for experiment_id in experiments_folders:
            experiment_path = os.path.join(self.mlruns_path, experiment_id)
            if os.path.isdir(experiment_path):
                experiment_name = self.get_experiment_name(experiment_path)
                if experiment_name is not None:
                    experiments[experiment_name] = experiment_path

        return experiments


    def results_per_run(self, run_path: str) -> tuple [dict, dict]:
        """
        Read the image-level or pixel-level metrics and selected tags from the run folder.
        """
        metrics_path = os.path.join(run_path, "metrics")
        tags_path = os.path.join(run_path, "tags")

        # read metrics
        results = {}
        for metric_file in os.listdir(metrics_path):
            metric_name = metric_file
            if metric_name.startswith("image_") or metric_name.startswith("pixel_"):
                with open(os.path.join(metrics_path, metric_file), 'r') as file:
                    results[metric_name] = file.read().split()[1]

        # read tags
        tags = {}
        for tag_file in os.listdir(tags_path):
            if tag_file == "model" or tag_file == "dataset" or tag_file == "cls" or tag_file == "seed":
                with open(os.path.join(tags_path, tag_file), 'r') as file:
                    tags[tag_file] = file.read().strip()

        return results, tags


    def summarize_results(
            self, 
            experiment_path: str, 
            columns: list = ["model", "dataset"]
        ) -> dict:
        """
        Summarize the results of the experiment by iterating
        over the runs and calculating the mean of the metrics.
        Args:
            experiment_path (str): Path to the experiment folder.
            columns (list): Columns to group the results by. Defaults to ["model", "dataset"].
        """
        # Prepare a list to collect run data
        runs_data = []

        # Create DataFrame with all tags and metrics as columns
        runs_folders = os.listdir(experiment_path)
        if "meta.yaml" in runs_folders:
            runs_folders.remove("meta.yaml")
        for run_id in runs_folders:
            results, tags = self.results_per_run(os.path.join(experiment_path, run_id))
            row = {"id": run_id}
            row.update(tags)
            row.update({k: float(v) for k, v in results.items()})
            runs_data.append(row)
        self.runs = pd.DataFrame(runs_data)

        # Group by the specified columns and calculate mean for each metric
        grouped = self.runs.groupby(columns)
        mean_df = grouped.mean(numeric_only=True).reset_index()

        # Convert metrics to percentage
        mean_df[mean_df.select_dtypes(include=['number']).columns] *= 100

        return mean_df.sort_values(by=columns)


    def print_metrics(self, experiment_names: None | list = None) -> None:
        """
        Print the metrics of all or selected experiments in a readable format.

        Args:
            experiment_names (list, optional): List of experiment names to print.
                If None, all experiments will be printed. Defaults to None.
        """
        experiments = self.iterate_experiments()

        if experiment_names is not None:
            if not isinstance(experiment_names, list):
                raise TypeError("Parameter 'experiment_names' must be a list.")
            names_set = set(experiment_names)
            experiments_set = set(experiments.keys())
            selected_experiments = list(names_set.intersection(experiments_set))
        else:
            selected_experiments = experiments.keys()

        for name in selected_experiments:
            path = experiments[name]

            print(f"Processing experiment: {name}")
            summary = self.summarize_results(path)
            print(summary)


    def plot_results_per_cathegory(
            self, 
            experiment_names: list, 
            model_name: str = "Patchcore",
            dataset_name: str = "Visa",
            metric: str = "image_AUROC",
            save_image: bool = False,
            save_path: str = "../visualizations"
        ) -> None:
        """
        Plot the results per category for the given metric, model and dataset.

        Args:
            experiment_names (list): List of experiment names to plot.
            model_name (str): Model name to filter results. Defaults to "Patchcore".
            dataset_name (str): Dataset name to filter results. Defaults to "Visa".
            metric (str): Metric to plot. Defaults to "image_AUROC".
            save_image (bool): If True, save the plot as an image. Defaults to False.
            save_path (str): Path to save the image if `save_image` is True. Defaults to "../visualizations".

        Raises:
            ValueError: If the dataset or model is not found in the results.
            ValueError: If the metric is not found in the results.
        """
        import matplotlib.pyplot as plt

        experiments = self.iterate_experiments()
        selected_experiments = [name for name in experiment_names if name in experiments]

        plt.figure(figsize=(12, 7))
        for name in selected_experiments:
            path = experiments[name]
            summary = self.summarize_results(path, columns=["model", "dataset", "cls"])

            dataset_summary = summary[summary["dataset"] == dataset_name]
            if dataset_summary.empty:
                raise ValueError(f"No results found for dataset '{dataset_name}' in experiment '{name}'.")
            
            dataset_summary = dataset_summary[dataset_summary["model"] == model_name]
            if dataset_summary.empty:
                raise ValueError(f"No results found for model '{model_name}' in experiment '{name}'.")
            
            if metric not in dataset_summary.columns:
                raise ValueError(f"Metric '{metric}' not found in the results for experiment '{name}'.")
            
            sorted_summary = dataset_summary.sort_values(by="cls")

            # Plot the metric for each category
            plt.plot(
                sorted_summary["cls"],
                sorted_summary[metric],
                marker='o',
                label=name,
                linewidth=2
            )

        plt.ylim(0, 100)
        plt.xlabel("Category")
        plt.ylabel(metric)
        plt.title(f"{metric} per category ({dataset_name}, {model_name})")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()

        if save_image:
            filename = f"{dataset_name}_{model_name}_{metric}.png"
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, filename), bbox_inches='tight', dpi=300)
        else:
            plt.show()
