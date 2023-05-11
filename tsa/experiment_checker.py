import dataclasses
from typing import Dict, Tuple, List

from mlflow.entities import RunStatus, Run

from tsa.experiment import Experiment, RunConfig


@dataclasses.dataclass()
class ExperimentStats:
    counts: Dict[RunStatus, Dict[int, int]]
    skipped: List[Run]
    missing_runs: List[RunConfig]
    duplicate_runs: List[Tuple[List[RunConfig], int]]


class ExperimentChecker:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment
        self.mlflow_client = experiment.mlflow

    def iter_mlflow_runs(self):
        exp = self.mlflow_client.get_experiment_by_name(self.experiment.name)
        if exp is None:
            raise RuntimeError("Experiment with name '%s' not found." % self.experiment.name)
        return self.mlflow_client.search_runs(experiment_ids=[exp.experiment_id], order_by=["start_time DESC"])

    def counts_by_run_status(self) -> Tuple[Dict[RunStatus, Dict[int, int]], List[Run]]:

        counts_by_status = {status: {} for status in RunStatus.all_status()}
        skipped = []
        for r in self.iter_mlflow_runs():
            status = RunStatus.from_string(r.info.status)
            if status not in counts_by_status:
                print(counts_by_status)
                raise ValueError("Unexpected status: %s" % r.info.status)
            if "iteration" not in r.data.params:
                print("Skip run %s, because it has no 'iteration' parameter" % r.info.run_id)
                skipped.append(r)
            iteration = int(r.data.params["iteration"])

            counts = counts_by_status[status]
            if iteration not in counts:
                counts[iteration] = 0
            counts[iteration] += 1

        return counts_by_status, skipped

    def stats(self) -> ExperimentStats:
        runs = self.experiment.run_configurations()
        counts, skipped = self.counts_by_run_status()

        fin_counts = counts[RunStatus.FINISHED]
        runn_counts = counts[RunStatus.RUNNING]

        missing = [run for run in runs if run.iteration not in fin_counts and run.iteration not in runn_counts]
        duplicates = [(run, fin_counts[run.iteration]) for run in runs if run.iteration if
                      run.iteration in fin_counts and fin_counts[run.iteration] > 1]
        # TODO add running runs duplicates
        return ExperimentStats(counts, skipped, missing, duplicates)

    def check_all(self):
        stats = self.stats()
        if len(stats.duplicate_runs) != 0:
            print("Duplicate runs:")
            for r, count in stats.duplicate_runs:
                print(f"{r}: {count}x")

        if len(stats.missing_runs) != 0:
            print("Missing Runs:")
            for r in stats.missing_runs:
                print(r)

    def next_free_iteration(self):
        runs = self.experiment.run_configurations()
        counts = {r.iteration: 0 for r in runs}
        exp = self.mlflow_client.get_experiment_by_name(self.experiment.name)
        if exp is None:
            raise RuntimeError("Experiment with name '%s' not found." % self.experiment.name)
        running_runs = []
        failed_runs = []
        for r in self.mlflow_client.search_runs(experiment_ids=[exp.experiment_id], order_by=["start_time DESC"]):
            if "iteration" not in r.data.params:
                print("Skip run %s, because it has no 'iteration' parameter" % r.info.run_id)
            iteration = int(r.data.params["iteration"])
            if r.info.status == "RUNNING":
                running_runs.append(iteration)
                continue
            if r.info.status == "FAILED":
                failed_runs.append(iteration)
                continue
            if r.info.status != "FINISHED":
                raise ValueError("Unexpected status: %s" % r.info.status)
            if iteration not in counts:
                raise ValueError("Unexpected iteration: %s" % iteration)
            counts[iteration] += 1

        max_i = 0
        for i, c in counts.items():
            if (c > 0 and i > max_i) or i in running_runs:
                max_i = i

        next_i = max_i + 1
        if next_i >= len(counts):
            raise ValueError("No free run found.")
        return next_i
