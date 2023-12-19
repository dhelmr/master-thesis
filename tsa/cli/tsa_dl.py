import tempfile
from argparse import ArgumentParser

import tempfile
from argparse import ArgumentParser

import mlflow
import pandas
from mlflow import MlflowClient

from tsa.cli.run import SubCommand, make_experiment_from_path
from tsa.experiment_checker import ExperimentChecker


class TSADownloaderSubCommand(SubCommand):

    def __init__(self):
        super().__init__("tsa-dl", "analyse training set experiments")
    def make_subparser(self, parser: ArgumentParser):
        parser.add_argument("--experiment", "-e", required=False, help="Sets the mlflow experiment ID", type=str)
        parser.add_argument("-c", "--config", required=True, help="Experiment config yaml file.")
        parser.add_argument("-o", "--output", required=True, help="Output file")
    def exec(self, args, parser, unknown_args):
        mlflow_client = MlflowClient() # TODO global singleton
        experiment = make_experiment_from_path(args.config, mlflow_client, args.experiment)
        checker = ExperimentChecker(experiment, no_ids_checks=True)
        dfs = []
        for run in checker.iter_mlflow_runs():
            artifacts = experiment.mlflow.list_artifacts(run.info.run_id)
            if len(artifacts) == 0:
                raise RuntimeError("No artifacts: %s" % run.info.run_id)
            for a in artifacts:
                if str(a.path).endswith(".parquet.gz"):
                    tmpfile = tempfile.mktemp()
                    print("Download...", run.info.run_id, a.path)
                    mlflow.artifacts.download_artifacts(run_id=run.info.run_id, artifact_path=a.path, dst_path=tmpfile)
                    # local_file = os.path.join(tmpfile, str(a.path))
                    df = pandas.read_parquet(tmpfile)
                    df["run_id"] = run.info.run_id
                    for col in run.data.params:
                        df[col] = run.data.params[col]
                    # df = df.append(run.data.params, ignore_index=True)
                    dfs.append(df)
        big_df = pandas.concat(dfs, axis=0)
        big_df.to_csv(args.output)
