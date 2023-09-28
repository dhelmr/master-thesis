import abc
import dataclasses
import itertools
from typing import List, Iterable

import numpy
import numpy as np
import pandas
import pandas as pd
from pandas import DataFrame

from tsa.confusion_matrix import ConfusionMatrix


class PerformancePredictor(abc.ABC):
    @abc.abstractmethod
    def train(self, train_X: pandas.DataFrame, train_y: numpy.ndarray):
        pass

    @abc.abstractmethod
    def predict(self, test_X: pandas.DataFrame) -> numpy.ndarray:
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
                 scenario_col="scenario"):
        self._df = df
        self._feature_cols = feature_cols
        self._scenario_col = scenario_col
        self.target_cols = [col for col in df.columns if col not in feature_cols and col != scenario_col]

    def feature_cols(self):
        return self._feature_cols

    def get_scenarios(self) -> List[str]:
        return pd.unique(self._df[self._scenario_col])

    def get_split(self, target_var: str, test_scenarios: List[str], threshold: float) -> TrainTestSplit:
        all_scenarios = self.get_scenarios()
        # validate parameters
        for sc in test_scenarios:
            if sc not in all_scenarios:
                raise ValueError("Scenario %s does not exist in the dataset" % sc)

        train_scenarios = [sc for sc in all_scenarios if sc not in test_scenarios]
        train = self._df[self._df[self._scenario_col].isin(train_scenarios)]
        test = self._df[self._df[self._scenario_col].isin(test_scenarios)]

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


class CV:
    def __init__(self, data: PerformanceData, predictor: PerformancePredictor, cv_leave_out=2):
        self.data = data
        self.cv_leave_out = cv_leave_out
        self.predictor = predictor

    def run(self, target_var: str, threshold: float):
        if target_var not in self.data.target_cols:
            raise ValueError("No target variable: %s" % target_var)
        all_metrics = []
        for split in self._iter_cv_splits(target_var, threshold):
            #if np.all(split.test_y == 0) or np.all(split.test_y == 1):
            #    print("Skip split, only one class in test data (test scenarios=%s)" % (split.test_scenarios, ))
            #    continue
            # TODO: preprocessing? (dim reduction, min-max scaling, ...?)
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
        stats = df.mean(numeric_only=True)
        return stats

    def _iter_cv_splits(self, target_var: str, threshold: float) -> Iterable[TrainTestSplit]:
        scenarios = self.data.get_scenarios()
        for val_scenarios in itertools.combinations(scenarios, self.cv_leave_out):
            split = self.data.get_split(target_var, val_scenarios, threshold)
            yield split


