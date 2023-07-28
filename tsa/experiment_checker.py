import dataclasses
import time
from datetime import datetime, timedelta
from typing import Dict, Tuple, List

import mlflow
from mlflow.entities import RunStatus, Run
from pandas import DataFrame

from tsa.experiment import Experiment, RunConfig, IGNORE_SCENARIOS


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

    def missing_but_not_running(self):
        running_iterations = set([c.iteration for c in self.missing_runs_but_running])
        return [c for c in self.missing_runs if c.iteration not in running_iterations]



class ExperimentChecker:
    def __init__(self, experiment: Experiment, no_ids_checks=False):
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
        filter_str = ""
        if not self.no_id_checks:
            filter_str = f"params.parameter_cfg_id = '{safe_filter_value(self.experiment.parameter_cfg_id)}'"

        token = None
        while True:
            result = self.mlflow_client.search_runs(experiment_ids=[exp.experiment_id],
                                                    order_by=["start_time DESC"],
                                                    filter_string=filter_str,
                                                    page_token=token)
            token = result.token
            for r in result:
                yield r
            if token is None:
                break

    def _mlflow_run_to_run_cfg(self, run: Run) -> RunConfig:
        if "iteration" not in run.data.params:
            raise ValueError("Could not find key 'iteration' in run config")
        iteration = int(run.data.params["iteration"])
        if "lid_ds_version" in run.data.params:
            scenario = run.data.params["lid_ds_version"] + "/" + run.data.params["scenario"]
        else:
            scenario = run.data.params["scenario"]
        return RunConfig(
            parameter_cfg_id=run.data.params["parameter_cfg_id"],
            num_attacks=int(run.data.params["num_attacks"]),
            iteration=iteration,
            permutation_i=int(run.data.params["permutation_i"]),
            scenario=scenario
        )

    def counts_by_run_status(self) -> Tuple[Dict[RunStatus, Dict[int, int]], List[Run], List[RunConfig]]:

        counts_by_status = {status: {} for status in RunStatus.all_status()}
        skipped = []
        parsed_runs = []
        for r in self.iter_mlflow_runs():
            status = RunStatus.from_string(r.info.status)
            if status not in counts_by_status:
                print(counts_by_status)
                raise ValueError("Unexpected status: %s" % r.info.status)

            try:
                run_cfg = self._mlflow_run_to_run_cfg(r)
            except Exception as e:
                print("Skip run %s, error is: %s" % (r.info.run_id, e))
                skipped.append(r)
                continue
            if run_cfg.scenario in IGNORE_SCENARIOS:
                continue
            counts = counts_by_status[status]
            if run_cfg.iteration not in counts:
                counts[run_cfg.iteration] = 0
            counts[run_cfg.iteration] += 1
            parsed_runs.append(run_cfg)

        return counts_by_status, skipped, parsed_runs

    def check_integrity(self, mlflow_run_cfgs: List[RunConfig], expected_run_cfgs: List[RunConfig]):
        failed_integrity = []
        not_found = []
        for mlflow_cfg in mlflow_run_cfgs:
            found_one = False
            for expected_run_cfg in expected_run_cfgs:
                if expected_run_cfg.iteration == mlflow_cfg.iteration:
                    if expected_run_cfg == mlflow_cfg:
                        found_one = True
                    else:
                        print(expected_run_cfg, mlflow_cfg)
                        failed_integrity.append(mlflow_cfg)
            if not found_one:
                not_found.append(mlflow_cfg)
        return failed_integrity, not_found

    def stats(self) -> ExperimentStats:
        runs = self.experiment.run_configurations()
        counts, skipped, parsed_run_cfgs = self.counts_by_run_status()

        failed_integrity, not_found = self.check_integrity(parsed_run_cfgs, runs)
        if len(failed_integrity) > 0:
            raise RuntimeError(
                "The following mlflow runs have unexpected parameters (based on their iteration id): %s" %
                [r.iteration for r in failed_integrity])
        if len(not_found) > 0:
            raise RuntimeError("The following mlflow runs are not expected %s" % [r.iteration for r in not_found])

        fin_counts = counts[RunStatus.FINISHED]
        runn_counts = counts[RunStatus.RUNNING]

        missing = [run for run in runs if run.iteration not in fin_counts]
        missing_but_running = [run for run in missing if run.iteration in runn_counts]
        duplicates = [(run, fin_counts[run.iteration]) for run in runs if run.iteration if
                      run.iteration in fin_counts and fin_counts[run.iteration] > 1]
        # TODO add running runs duplicates
        return ExperimentStats(counts, runs, skipped, missing, missing_but_running, duplicates)

    def get_stale_runs(self, older_than: timedelta):
        now = time.time() * 1000
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

    def get_runs_df(self, no_finished_check=False) -> Tuple[DataFrame, bool]:
        stats = self.stats()
        if len(stats.skipped) != 0:
            raise RuntimeError("Skipped some runs: %s " % stats.skipped)
        if len(stats.duplicate_runs) != 0:
            raise RuntimeError("Found duplicate runs: %s" % stats.duplicate_runs)
        if not stats.is_finished():
            if no_finished_check:
                print("WARNING: experiment is not finished!")
            else:
                raise RuntimeError("Experiment is not finished.")
        exp = mlflow.get_experiment_by_name(self.experiment.mlflow_name)
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        runs = runs.loc[~runs["params.dataloader.scenario"].isin(IGNORE_SCENARIOS)]
        return runs, stats.is_finished()


def safe_filter_value(value: str) -> str:
    return value.replace("'", "\\'")

