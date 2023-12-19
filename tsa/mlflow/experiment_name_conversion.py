import dataclasses
import datetime
import os
from typing import Optional

import pandas
from pandas import DataFrame


class ExperimentNameConversion:
    def __init__(self):
        if "EXPERIMENT_PREFIX" not in os.environ:
            raise KeyError("$EXPERIMENT_PREFIX is not set")
        self.mlflow_exp_prefix = os.environ["EXPERIMENT_PREFIX"]
        if "EXPERIMENT_BASE_PATH" not in os.environ:
            raise KeyError("$EXPERIMENT_BASE_PATH is not set")
        self.exp_base_path = os.path.abspath(os.environ["EXPERIMENT_BASE_PATH"])

    def infer_exp_name(self, config_path: str):
        rel_exp_path = os.path.abspath(config_path)[len(self.exp_base_path) + 1 :]
        exp_name = "%s/%s" % (self.mlflow_exp_prefix, rel_exp_path.replace("/", "-"))
        return exp_name

    def get_rel_exp_name(self, mlflow_name: str):
        return mlflow_name[len(self.mlflow_exp_prefix) + 1 :]


@dataclasses.dataclass
class CachedResult:
    df: DataFrame
    timestamp: datetime.datetime


class MlflowResultsCache:
    def __init__(self, cache_path: str):
        self._cache_path = cache_path

    def get_result_path(self, exp_name: str):
        return os.path.join(self._cache_path, exp_name + ".csv")

    def get_cached_result(self, exp_name: str) -> Optional[CachedResult]:
        result_file = self.get_result_path(exp_name)
        if not os.path.exists(result_file):
            return None
        epoch_time = os.path.getmtime(result_file)
        df = pandas.read_csv(result_file)
        return CachedResult(
            df=df, timestamp=datetime.datetime.utcfromtimestamp(epoch_time)
        )

    def cache(self, exp_name: str, df: DataFrame):
        result_file = self.get_result_path(exp_name)
        df.to_csv(result_file, index=False)
