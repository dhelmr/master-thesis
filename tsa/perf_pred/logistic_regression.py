import argparse

import numpy
import pandas
import sklearn
from sklearn import linear_model

from tsa.perf_pred.cv import PerformancePredictor


class LogisticRegression(PerformancePredictor):
    def __init__(self, cli_args=[]):
        parser = argparse.ArgumentParser()
        parser.add_argument("--penalty", default="l2", type=str)
        args = parser.parse_args(cli_args)
        self.clf_args = vars(args)
        self.reset()

    def train(self, train_X: pandas.DataFrame, train_y: numpy.ndarray):
        train_set = train_X.values
        # X_train_scaled = self.prepr_pl.fit_transform(train_set)
        X_train_scaled = train_set
        self.feature_names = list(train_X.columns)
        self.clf.fit(X_train_scaled, train_y)

    def predict(self, test_X: pandas.DataFrame) -> numpy.ndarray:
        # X_test_scaled = self.prepr_pl.transform(test_X.values)
        X_test_scaled = test_X.values
        preds = self.clf.predict(X_test_scaled)
        return preds

    def reset(self):
        self.clf = sklearn.linear_model.LogisticRegression(
            random_state=1, **self.clf_args
        )
        # self.prepr_pl = Pipeline([("min-max", MinMaxScaler())])
        # self.prepr_pl = Pipeline([])
        self.feature_names = []

    def extract_rules(self, out_path: str, class_names=["0", "1"]):
        return "%s intercept: %s" % (
            (list(zip(self.feature_names, self.clf.coef_.tolist()[0]))),
            self.clf.intercept_,
        )
