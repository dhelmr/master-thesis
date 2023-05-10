from tsa.experiment import Experiment, RunConfig


class ExperimentChecker:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment
        self.mlflow_client = experiment.mlflow

    def check_all(self, experiment_name: str):
        runs = self.experiment.run_configurations()
        counts = {r.iteration: 0 for r in runs}
        exp = self.mlflow_client.get_experiment_by_name(experiment_name)
        if exp is None:
            raise RuntimeError("Experiment with name '%s' not found." % experiment_name)
        running_runs = []
        for r in self.mlflow_client.search_runs(experiment_ids=[exp.experiment_id], order_by=["start_time DESC"]):
            if "iteration" not in r.data.params:
                print("Skip run %s, because it has no 'iteration' parameter" % r.info.run_id)
            if r.info.status == "RUNNING":
                running_runs.append(r)
                continue
            # TODO FAILED runs
            iteration = int(r.data.params["iteration"])
            if iteration not in counts:
                raise ValueError("Unexpected iteration: %s" % iteration)
            counts[iteration] += 1

        missing = [iteration for iteration, count in counts.items() if count == 0]
        duplicates = [iteration for iteration, count in counts.items() if count > 1]

        if len(missing) != 0:
            print("Missing Runs:")
            missing_runs = [run_cfg for run_cfg in runs if run_cfg.iteration in missing]
            for r in missing_runs:
                currently_running = [cr for cr in running_runs if cr.data.params["iteration"] == str(r.iteration)]
                print(r)

        if len(duplicates) != 0:
            print("Duplicate runs:")
            d_runs = [run_cfg for run_cfg in runs if run_cfg.iteration in duplicates]
            for r in d_runs:
                print(f"{r}: {counts[r.iteration]}x")

        print("Currently running:", [r.data.params["iteration"] for r in running_runs])

    def next_free_iteration(self, experiment_name: str):
        runs = self.experiment.run_configurations()
        counts = {r.iteration: 0 for r in runs}
        exp = self.mlflow_client.get_experiment_by_name(experiment_name)
        if exp is None:
            raise RuntimeError("Experiment with name '%s' not found." % experiment_name)
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

        next_i = max_i+1
        if next_i >= len(counts):
            raise ValueError("No free run found.")
        return next_i