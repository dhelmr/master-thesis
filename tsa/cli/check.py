from argparse import ArgumentParser
from datetime import timedelta, datetime
from time import localtime, strftime

import yaml
from mlflow import MlflowClient
from mlflow.entities import RunStatus

from tsa.cli.run import SubCommand, make_experiment_from_path, make_experiment_from_mlflow
from tsa.cli.search import ParameterSearch
from tsa.experiment_checker import ExperimentChecker, ExperimentStats
from tsa.mlflow.experiment_name_conversion import ExperimentNameConversion


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class CheckSubCommand(SubCommand):

    def __init__(self):
        super().__init__("check", "check experiment for completeness and integrity")
        self._no_choice = False

    def make_subparser(self, parser: ArgumentParser):
        parser.add_argument("--experiment", "-e", required=False,
                            help="Sets the mlflow experiment name. If not set, it will be inferred from the config path.",
                            type=str)
        parser.add_argument("-c", "--config", required=True, help="Experiment config yaml file.")
        parser.add_argument("--remove-stale", help="remove stale runs (RUNNING but older than x hours)",
                            action="store_true")
        parser.add_argument("--stale-hours", help="defines after how many hours RUNNING runs defined as 'stale'",
                            type=int, default=49)
        parser.add_argument("--remove-duplicate", action="store_true", default=False)
        parser.add_argument("--no-choice", action="store_true", default=False, help="Assume yes/first choice if asked for confirmation. (used for remove-duplicates or remove-stale)")
        parser.add_argument("--verbose", action="store_true", help="Verbose output", default=False)

    def exec(self, args, parser):
        self._no_choice = args.no_choice
        if args.experiment is not None:
            experiment_name = args.experiment
        else:
            converter = ExperimentNameConversion()
            experiment_name = converter.infer_exp_name(args.config)

        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        if "experiment" in cfg:
            # experiment config is a search experiment
            search = ParameterSearch(cfg)
            mlflow = MlflowClient()
            checkers = search.iterate_exp_checkers(mlflow, experiment_name)
        else:
            exp = load_exp_from_parser(args.config, experiment_name)
            checkers = [exp]
        count_ok = 0
        count_not_ok = 0
        for checker in checkers:
            print("===> parameter_cfg_id: %s" % checker.experiment.parameter_cfg_id)
            if args.remove_stale:
                self._remove_stale(args, checker)
            elif args.remove_duplicate:
                stats = checker.stats()
                self._remove_duplicate(stats, checker)
            else:
                stats = checker.stats()
                ok, status = self.print_stats(stats, verbose=args.verbose)
                if ok:
                    print(bcolors.OKGREEN + "Ok" + bcolors.ENDC)
                    count_ok += 1
                else:
                    print(bcolors.WARNING + "NOT OK: %s" % status + bcolors.ENDC)
                    count_not_ok += 1
            print("====================================================")

        if count_ok + count_not_ok > 1:
            if count_ok > 0:
                print(bcolors.OKGREEN + "OK: %s" % count_ok + bcolors.ENDC)
            if count_not_ok > 0:
                print(bcolors.WARNING + "NOT OK: %s" % count_not_ok + bcolors.ENDC)

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
        if self.choice("Remove these %s runs?" % len(stale)) == "y":
            for s in stale:
                checker.experiment.mlflow.delete_run(s.info.run_id)

    def _print_duplicate_runs(self, stats):
        print("Duplicate runs:")
        index = 0
        for r, count in stats.duplicate_runs:
            print(f"[{index}] {r}: {count}x")

    def print_stats(self, stats, verbose=False):
        ok = True
        status = ""

        if len(stats.counts[RunStatus.RUNNING]) != 0:
            if verbose:
                print("RUNNING runs:")
                for i, count in stats.counts[RunStatus.RUNNING].items():
                    r = stats.run_configs[i]
                    print(r, f"({count}x)")
            ok = False
            status += "%s running runs; " % len(stats.counts[RunStatus.RUNNING])

        if len(stats.duplicate_runs) != 0:
            self._print_duplicate_runs(stats)
            ok = False
            status += "%s duplicate runs; " % len(stats.duplicate_runs)

        if len(stats.missing_runs) != 0:
            if verbose:
                print("Missing Runs:")
                for r in stats.missing_runs:
                    if r in stats.missing_runs_but_running:
                        print(r, "(RUNNING)")
                    else:
                        print(r)
            ok = False
            status += "%s missing runs; " % len(stats.missing_runs)

        finished_count = len(stats.run_configs) - len(stats.missing_runs)
        print("Progress: %s finished; %s running (out of missing runs)" %
              (format_perc(finished_count, len(stats.run_configs)),
               format_perc(len(stats.missing_runs_but_running), len(stats.missing_runs))
               ))
        return ok, status

    def _remove_duplicate(self, stats: ExperimentStats, checker):
        if len(stats.duplicate_runs) == 0:
            return
        for run_cfg, count in stats.duplicate_runs:
            found_runs = []
            for r in checker.iter_mlflow_runs():
                if int(r.data.params["iteration"]) != int(run_cfg.iteration):
                    continue
                found_runs.append(r)
            for i, r in enumerate(found_runs):
                print("[%s] %s - %s - %s" % (i, r.info.run_name, r.info.status, r.info.start_time))
            keep = self.choice("Found %s mlflow runs for run cfg %s. Enter which run to keep. (k for keep all)" % (
            len(found_runs), run_cfg),
                          choices=[str(index) for index in range(len(found_runs))] + ["k"])
            if keep == "k":
                continue
            else:
                for i, r in enumerate(found_runs):
                    if i == int(keep):
                        continue
                    print("Delete", r.info.run_name)
                    checker.experiment.mlflow.delete_run(r.info.run_id)


    def choice(self, msg: str, choices=["y", "n"]):
        if self._no_choice:
            return choices[0]
        inp = None
        while inp not in choices:
            inp = input("%s [%s]" % (msg, ", ".join(choices)))
        return inp


def load_exp_from_parser(config, experiment_name):
    mlflow_client = MlflowClient()  # TODO global singleton
    if config is None:
        experiment = make_experiment_from_mlflow(mlflow_client, experiment_name)
    else:
        experiment = make_experiment_from_path(config, mlflow_client, experiment_name)
    checker = ExperimentChecker(experiment, no_ids_checks=True)
    return checker


def format_perc(fraction, of):
    if of == 0:
        return "%s/%s" % (fraction, of)
    return "%s/%s (%s perc.)" % (
        fraction, of, fraction / of * 100
    )
