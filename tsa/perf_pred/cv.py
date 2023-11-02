import abc
import dataclasses
import itertools
from typing import List, Iterable, Any, Dict

import numpy
import numpy as np
import pandas
import pandas as pd
from pandas import DataFrame

from tsa.confusion_matrix import ConfusionMatrix


class PerformancePredictor(abc.ABC):

    def __init__(self, cli_args):
        pass

    def reset(self):
        pass

    @abc.abstractmethod
    def train(self, train_X: pandas.DataFrame, train_y: numpy.ndarray):
        pass

    @abc.abstractmethod
    def predict(self, test_X: pandas.DataFrame) -> numpy.ndarray:
        pass

    def extract_rules(self):
        pass


@dataclasses.dataclass
class TrainTestSplit:
    test_scenarios: List[str]
    train_scenarios: List[str]
    target_var: str
    threshold: float
    train_X: pandas.DataFrame
    train_y: np.ndarray
    test_X: pandas.DataFrame
    test_y: np.ndarray


class PerformanceData:
    def __init__(self, df: DataFrame,
                 feature_cols: List[str],
                 scenario_col="scenario",
                 syscalls_col="syscalls"):
        self.df = df
        self._feature_cols = feature_cols
        self.scenario_col = scenario_col
        self.target_cols = [col for col in df.columns if col not in feature_cols and col != scenario_col]
        self.syscalls_col = syscalls_col

    def feature_cols(self):
        return self._feature_cols

    def get_scenarios(self) -> List[str]:
        return pd.unique(self.df[self.scenario_col])

    def get_scenario_data(self, sc_name: str) -> pandas.DataFrame:
        return self.df.loc[self.df[self.scenario_col] == sc_name]

    def get_split(self, target_var: str, test_scenarios: List[str], threshold: float) -> TrainTestSplit:
        all_scenarios = self.get_scenarios()
        # validate parameters
        for sc in test_scenarios:
            if sc not in all_scenarios:
                raise ValueError("Scenario %s does not exist in the dataset" % sc)

        train_scenarios = [sc for sc in all_scenarios if sc not in test_scenarios]
        train = self.df[self.df[self.scenario_col].isin(train_scenarios)]
        test = self.df[self.df[self.scenario_col].isin(test_scenarios)]

        train_X = train.filter(self._feature_cols)
        test_X = test.filter(self._feature_cols)

        map_to_binary_class = lambda value: 1 if value > threshold else 0
        train_y = train[target_var].apply(map_to_binary_class).to_numpy()
        test_y = test[target_var].apply(map_to_binary_class).to_numpy()

        return TrainTestSplit(
            target_var=target_var,
            test_scenarios=test_scenarios,
            train_scenarios=train_scenarios,
            threshold=threshold,
            test_y=test_y,
            test_X=test_X,
            train_y=train_y,
            train_X=train_X
        )

    def with_features(self, feature_cols: List[str]) -> "PerformanceData":
        return PerformanceData(
            self.df,
            feature_cols=feature_cols,
            scenario_col=self.scenario_col,
            syscalls_col=self.syscalls_col
        )


CVPerformance = Any


class CV:
    def __init__(self, data: PerformanceData, predictor: PerformancePredictor, cv_leave_out=2):
        self.data = data
        self.cv_leave_out = cv_leave_out
        self.predictor = predictor

    def run(self, target_var: str, threshold: float) -> CVPerformance:
        if target_var not in self.data.target_cols:
            raise ValueError("No target variable: %s" % target_var)
        all_metrics = []
        for split in self._iter_cv_splits(target_var, threshold):
            # if np.all(split.test_y == 0) or np.all(split.test_y == 1):
            #    print("Skip split, only one class in test data (test scenarios=%s)" % (split.test_scenarios, ))
            #    continue
            # TODO: preprocessing? (dim reduction, min-max scaling, ...?)
            self.predictor.reset()
            self.predictor.train(split.train_X, split.train_y)
            preds = self.predictor.predict(split.test_X)
            cm = ConfusionMatrix.from_predictions(preds, split.test_y, labels=[0, 1])
            metrics = cm.calc_unweighted_measurements()
            row_data = {
                "test_scenarios": split.test_scenarios,
                "train_scenarios": split.train_scenarios,
                "target_var": split.target_var,
                "threshold": split.threshold,
                **metrics
            }
            all_metrics.append(row_data)
        df = pandas.DataFrame(all_metrics)
        # todo?
        df.fillna(value=0, inplace=True)
        mean_stats = df.mean(numeric_only=True)
        var_stats = df.var(numeric_only=True)
        min_stats = df.min(numeric_only=True)
        max_stats = df.max(numeric_only=True)
        range_stats = max_stats - min_stats
        stats = pandas.concat([mean_stats, var_stats, min_stats, max_stats, range_stats], axis=1).rename(columns={0: "mean", 1: "var", 2: "min", 3: "max", 4: "range"})
        return unpack_dict(stats.to_dict())

    def _iter_cv_splits(self, target_var: str, threshold: float) -> Iterable[TrainTestSplit]:
        scenarios = self.data.get_scenarios()
        for val_scenarios in itertools.combinations(scenarios, self.cv_leave_out):
            split = self.data.get_split(target_var, val_scenarios, threshold)
            yield split

def unpack_dict(d: Dict[str, Any], key_prefix="", to_dict = None):
    if to_dict is None:
        to_dict = {}
    for k in d.keys():
        if isinstance(d[k], dict):
            unpack_dict(d[k], key_prefix=f"{key_prefix}{k}.", to_dict=to_dict)
        else:
            to_dict[f"{key_prefix}{k}"] = d[k]
    return to_dict