import argparse

import numpy
import numpy as np
import pandas
from sklearn.ensemble import RandomForestClassifier

from tsa.perf_pred.cv import PerformancePredictor


class RandomForrest(PerformancePredictor):

    def __init__(self, cli_args=[]):
        parser = argparse.ArgumentParser()
        parser.add_argument("--max-depth", default=5, type=int)
        #parser.add_argument("--min-samples-leaf", default=3, type=int)
        args = parser.parse_args(cli_args)
        self.clf_args = vars(args)

        self.reset()

    def train(self, train_X: pandas.DataFrame, train_y: numpy.ndarray):
        train_X = self._preprocess(train_X)
        train_set = train_X.values
        #X_train_scaled = self.prepr_pl.fit_transform(train_set)
        X_train_scaled = train_set
        self.feature_names = list(train_X.columns)
        self.clf.fit(X_train_scaled, train_y)

    def predict(self, test_X: pandas.DataFrame) -> numpy.ndarray:
        test_X = self._preprocess(test_X)
        #X_test_scaled = self.prepr_pl.transform(test_X.values)
        X_test_scaled = test_X.values
        preds = self.clf.predict(X_test_scaled)
        return preds

    def _preprocess(self, df):
        df = df.replace([np.inf, -np.inf], np.nan)
        return df.fillna(0)
    def reset(self):
        self.clf = RandomForestClassifier(random_state=1, **self.clf_args)
        # self.prepr_pl = Pipeline([("min-max", MinMaxScaler())])
        #self.prepr_pl = Pipeline([])
        self.feature_names=[]

    def extract_rules(self, out_path: str, class_names = ["0", "1"]):
        #tree_rules = export_text(self.clf, feature_names=self.feature_names, decimals=6)
        #dt_to_svg(self.clf, feature_names=self.feature_names, target_names=class_names, out_path=out_path)
        return None

