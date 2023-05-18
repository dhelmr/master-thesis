from argparse import ArgumentParser
from datetime import timedelta, datetime
from time import localtime, strftime

import mlflow.runs
from mlflow import MlflowClient

from tsa.cli.run import SubCommand, make_experiment_from_path
from tsa.experiment_checker import ExperimentChecker


class CheckSubCommand(SubCommand):

    def __init__(self):
        super().__init__("check", "check experiment for completeness and integrity")
    def make_subparser(self, parser: ArgumentParser):
        parser.add_argument("--experiment", "-e", required=False, help="Sets the mlflow experiment ID", type=str)
        parser.add_argument("-c", "--config", required=True, help="Experiment config yaml file.")

        parser.add_argument("--remove-stale", help="remove stale runs (RUNNING but older than x hours)", action="store_true")
        parser.add_argument("--stale-hours", help="defines after how many hours RUNNING runs defined as 'stale'", type=int, default=49)

    def exec(self, args, parser):
        mlflow_client = MlflowClient() # TODO global singleton
        experiment = make_experiment_from_path(args.config, mlflow_client, args.experiment)
        checker = ExperimentChecker(experiment, no_ids_checks=True)
        if args.remove_stale:
            self._remove_stale(args, checker)
        else:
            checker.check_all()

    def _remove_stale(self, args, checker: ExperimentChecker):
        older_than = timedelta(hours=args.stale_hours)
        stale = checker.get_stale_runs(older_than)
        if len(stale) == 0:
            print("No stale runs found.")
            return
        print("Found %s stale runs (state=RUNNING but start_time older than %s hours)" % (len(stale), args.stale_hours))
        for s in stale:
            start_time_seconds = int(s.info.start_time / 1000)
            start_time = strftime('%Y-%m-%d %H:%M:%S', localtime(start_time_seconds))
            print(s.info.run_id, start_time)
        if yes_no("Remove these %s runs?" % len(stale)) == "y":
            for s in stale:
                checker.experiment.mlflow.delete_run(s.info.run_id)
def yes_no(msg: str, choices = ["y", "n"]):
    inp = None
    while inp not in choices:
        inp = input("%s [%s]" % (msg, ", ".join(choices)))
    return inp
