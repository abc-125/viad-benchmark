import os
import pandas as pd
import yaml


class MetricsReader():
    """
    Class for reading MLFlow metrics.
    """
    def __init__(self, mlruns_path="./mlruns", experiment_name=None):
        self.mlruns_path = mlruns_path
        self.experiment_name = experiment_name
        self.runs = {}

    def get_experiment_name(self, experiment_path):
        """
        Get the experiment name from the meta.yaml file in the experiment folder.
        """
        meta_file = os.path.join(experiment_path, "meta.yaml")
        if os.path.exists(meta_file):
            with open(meta_file, 'r') as f:
                meta_data = yaml.safe_load(f)
                return meta_data.get("name", "unknown_experiment")
        return "unknown_experiment"


    def iterate_experiments(self):
        """
        Iterate over the experiments in the mlruns folder 
        and return a dictionary of experiment names and paths.
        Returns a list of experiment names if experiment_name is not provided. 
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
                experiments[experiment_name] = experiment_path

                # return specific experiment if requested
                if self.experiment_name is not None:
                    return {self.experiment_name: experiment_path}
            
        return experiments


    def results_per_run(self, run_path):
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


    def summarize_results(self, experiment_path):
        """
        Summarize the results of the experiment by iterating 
        over the runs and calculating the mean of the metrics.
        """
        # iterate over runs in the experiment
        runs_folders = os.listdir(experiment_path)
        runs_folders.remove("meta.yaml")
        for run_id in runs_folders:
            results, tags = self.results_per_run(os.path.join(experiment_path, run_id))
            self.runs[run_id] = (results, tags)

        # calculate the mean of the metrics
        summary = {}
        for run_id, (results, tags) in self.runs.items():
            model_name = tags.get("model", "unknown_model")
            dataset_name = tags.get("dataset", "unknown_dataset")
            key = (model_name, dataset_name)
            if key not in summary:
                summary[key] = {}

            for metric, value in results.items():
                value = float(value)*100  # convert to percentage
                if metric not in summary[key]:
                    summary[key][metric] = []
                summary[key][metric].append(value)

        # calculate the mean for each metric
        for key, metrics in summary.items():
            for metric, values in metrics.items():
                summary[key][metric] = sum(values) / len(values)

        return summary


    def print_metrics(self):
        """
        Print the metrics of all experiments in a readable format.
        """
        experiments = self.iterate_experiments()

        for experiment_name, experiment_path in experiments.items():
            print(f"Processing experiment: {experiment_name}")
            summary = self.summarize_results(experiment_path)

            # convert summary to a DataFrame for better readability
            summary_df = pd.DataFrame.from_dict(summary, orient="index")
            summary_df.index = pd.MultiIndex.from_tuples(summary_df.index, names=["model", "dataset"])
            print(summary_df)
