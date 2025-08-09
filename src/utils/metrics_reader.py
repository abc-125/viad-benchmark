import os
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import itertools
import numpy as np


class MetricsReader():
    """
    Class for reading MLFlow metrics.
    """
    def __init__(
            self, 
            mlruns_path="./mlruns",
            metrics=["image_AUROC", "pixel_AUPRO"],
            models=["Patchcore", "ReverseDistillation", "Fastflow", "glass"],
        ):
        self.mlruns_path = mlruns_path
        self.metrics = metrics
        self.models = models
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
            if metric_file not in self.metrics:
                continue
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

        # Remove models that are not in the list of models
        self.runs = self.runs[self.runs["model"].isin(self.models)]

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
            print(summary.to_markdown())


    def plot_results_per_category(
            self, 
            experiment_names: list, 
            model_names: list = ["Patchcore"],
            dataset_name: str = "Visa",
            metric: str = "image_AUROC",
            save_image: bool = False,
            save_path: str = "../visualizations"
        ) -> None:
        """
        Plot the results per category for the given metric, models, and dataset.

        Args:
            experiment_names (list): List of experiment names to plot.
            model_names (list): List of model names to filter results. Defaults to ["Patchcore"].
            dataset_name (str): Dataset name to filter results. Defaults to "Visa".
            metric (str): Metric to plot. Defaults to "image_AUROC".
            save_image (bool): If True, save the plot as an image. Defaults to False.
            save_path (str): Path to save the image if `save_image` is True. Defaults to "../visualizations".

        Raises:
            ValueError: If the dataset, model or metric is not found in the results.
        """

        experiments = self.iterate_experiments()
        selected_experiments = [name for name in experiment_names if name in experiments]

        # Assign a color to each model
        model_names_full_list = ["Patchcore", "ReverseDistillation", "Fastflow", "glass"]
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        model_colors = {model: color_cycle[i % len(color_cycle)] for i, model in enumerate(model_names_full_list)}

        # Define line styles for each experiment
        line_styles = ['-', '--', ':']
        marker_styles = ['x']
        style_combinations = list(itertools.product(line_styles, marker_styles))

        plt.figure(figsize=(12, 7))
        style_idx = 0
        for name in selected_experiments:
            path = experiments[name]
            summary = self.summarize_results(path, columns=["model", "dataset", "cls"])

            dataset_summary = summary[summary["dataset"] == dataset_name]
            if dataset_summary.empty:
                raise ValueError(f"No results found for dataset '{dataset_name}' in experiment '{name}'.")

            for model_name in model_names:
                model_summary = dataset_summary[dataset_summary["model"] == model_name]
                if model_summary.empty:
                    raise ValueError(f"No results found for model '{model_name}' in experiment '{name}'.")
                if metric not in model_summary.columns:
                    raise ValueError(f"Metric '{metric}' not found in the results for experiment '{name}'.")
                sorted_summary = model_summary.sort_values(by="cls")

                # Assign color by model, line/marker style by experiment
                color = model_colors[model_name]
                line_style, marker = style_combinations[style_idx % len(style_combinations)]

                plt.plot(
                    sorted_summary["cls"],
                    sorted_summary[metric],
                    label=f"{name} - {model_name}",
                    color=color,
                    linestyle=line_style,
                    marker=marker,
                    linewidth=2
                )
            style_idx += 1

        plt.ylim(0, 100)
        plt.xlabel("Category")
        plt.ylabel(metric)
        plt.title(f"{metric} per category ({dataset_name})")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()

        if save_image:
            filename = f"{dataset_name}_{'_'.join(model_names)}_{metric}.png"
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, filename), bbox_inches='tight', dpi=300)
        else:
            plt.show()


    def plot_star_chart_difference(
            self, 
            experiment1: str = "eval_rgb",
            experiment2: str = "eval_grayscale_1ch", 
            dataset_name: str = "Visa",
            metric: str = "image_AUROC",
        ) -> None:
        """
        Create a star chart to compare the difference in metrics for two experiments.
        Args:
            experiment1 (str): Name of the first experiment.
            experiment2 (str): Name of the second experiment.
            dataset_name (str): Name of the dataset to filter results.
            metric (str): Metric to compare.

        Raises:
            ValueError: If one or both experiment names are not found.
        """

        # Get experiments
        experiments = self.iterate_experiments()
        if experiment1 not in experiments or experiment2 not in experiments:
            raise ValueError("One or both experiment names not found.")

        # Prepare data for both experiments
        summary1 = self.summarize_results(experiments[experiment1], columns=["model", "dataset", "cls"])
        summary2 = self.summarize_results(experiments[experiment2], columns=["model", "dataset", "cls"])

        # Filter by dataset and metric
        summary1 = summary1[summary1["dataset"] == dataset_name]
        summary2 = summary2[summary2["dataset"] == dataset_name]

        classes = sorted(set(summary1["cls"]).intersection(set(summary2["cls"])))
        num_classes = len(classes)
        angles = np.linspace(0, 2 * np.pi, num_classes, endpoint=False).tolist()
        angles += angles[:1]  # close the loop

        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)

        # Add a thicker gray line for zero
        ax.plot(angles, [0] * len(angles), color='gray', linewidth=2.5, linestyle='--', zorder=0)

        for model in self.models:
            model1 = summary1[summary1["model"] == model].set_index("cls")
            model2 = summary2[summary2["model"] == model].set_index("cls")
            if model1.empty or model2.empty:
                continue

            values1 = [model1.loc[cls, metric] if cls in model1.index else 0 for cls in classes]
            values2 = [model2.loc[cls, metric] if cls in model2.index else 0 for cls in classes]
            diff = [v1 - v2 for v1, v2 in zip(values1, values2)]
            diff += diff[:1]  # close the loop

            ax.plot(angles, diff, label=model, marker='o')
            ax.fill(angles, diff, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(classes)
        ax.set_title(f"Difference in {metric} per category ({dataset_name})\n{experiment1} - {experiment2}")
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        plt.tight_layout()
        plt.show()
