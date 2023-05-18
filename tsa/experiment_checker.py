import dataclasses
import time
from datetime import datetime, timedelta
from typing import Dict, Tuple, List

from mlflow.entities import RunStatus, Run

from tsa.experiment import Experiment, RunConfig


@dataclasses.dataclass()
class ExperimentStats:
    counts: Dict[RunStatus, Dict[int, int]]
    run_configs: List[RunConfig]
    skipped: List[Run]
    missing_runs: List[RunConfig]
    missing_runs_but_running: List[RunConfig]
    duplicate_runs: List[Tuple[RunConfig, int]]

    def is_finished(self):
        return len(self.missing_runs) == 0 and len(self.missing_runs_but_running) == 0


class ExperimentChecker:
    def __init__(self, experiment: Experiment, no_ids_checks = False):
        self.experiment = experiment
        self.mlflow_client = experiment.mlflow
        self.no_id_checks = no_ids_checks

    def exists_in_mlflow(self) -> bool:
        exp = self.mlflow_client.get_experiment_by_name(self.experiment.mlflow_name)
        return exp is not None

    def iter_mlflow_runs(self):
        exp = self.mlflow_client.get_experiment_by_name(self.experiment.mlflow_name)
        if exp is None:
            raise RuntimeError("Experiment with name '%s' not found." % self.experiment.mlflow_name)
        for r in self.mlflow_client.search_runs(experiment_ids=[exp.experiment_id], order_by=["start_time DESC"]):
            if self.no_id_checks or r.data.params["parameter_cfg_id"] == self.experiment.parameter_cfg_id:
                yield r

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

        missing = [run for run in runs if run.iteration not in fin_counts]
        missing_but_running = [run for run in missing if run.iteration in runn_counts]
        duplicates = [(run, fin_counts[run.iteration]) for run in runs if run.iteration if
                      run.iteration in fin_counts and fin_counts[run.iteration] > 1]
        # TODO add running runs duplicates
        return ExperimentStats(counts, runs, skipped, missing, missing_but_running, duplicates)

    def check_all(self):
        stats = self.stats()

        if len(stats.counts[RunStatus.RUNNING]) != 0:
            print("RUNNING runs:")
            for i, count in stats.counts[RunStatus.RUNNING].items():
                r = stats.run_configs[i]
                print(r, f"({count}x)")

        if len(stats.duplicate_runs) != 0:
            print("Duplicate runs:")
            for r, count in stats.duplicate_runs:
                print(f"{r}: {count}x")

        if len(stats.missing_runs) != 0:
            print("Missing Runs:")
            for r in stats.missing_runs:
                if r in stats.missing_runs_but_running:
                    print(r, "(RUNNING)")
                else:
                    print(r)

    def get_stale_runs(self, older_than: timedelta):
        now = time.time()*1000
        stale_runs = []
        for r in self.iter_mlflow_runs():
            status = RunStatus.from_string(r.info.status)
            time_elapsed = timedelta(milliseconds=now - r.info.start_time)
            if status == RunStatus.RUNNING and time_elapsed > older_than:
                stale_runs.append(r)
        return stale_runs

    def next_free_iteration(self):
        runs = self.experiment.run_configurations()
        counts = {r.iteration: 0 for r in runs}
        exp = self.mlflow_client.get_experiment_by_name(self.experiment.mlflow_name)
        if exp is None:
            raise RuntimeError("Experiment with name '%s' not found." % self.experiment.mlflow_name)
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
