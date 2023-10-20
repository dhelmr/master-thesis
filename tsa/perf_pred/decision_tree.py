import argparse

import matplotlib.pyplot as plt
import numpy
import pandas
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from tsa.perf_pred.cv import PerformancePredictor

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from sklearn.tree import export_text


class DecisionTree(PerformancePredictor):

    def __init__(self, cli_args=[]):
        parser = argparse.ArgumentParser()
        parser.add_argument("--max-depth", default=3, type=int)
        args = parser.parse_args(cli_args)
        self.clf_args = vars(args)

        self.reset()

    def train(self, train_X: pandas.DataFrame, train_y: numpy.ndarray):
        train_set = train_X.values
        #X_train_scaled = self.prepr_pl.fit_transform(train_set)
        X_train_scaled = train_set
        self.feature_names = list(train_X.columns)
        self.clf.fit(X_train_scaled, train_y)

    def predict(self, test_X: pandas.DataFrame) -> numpy.ndarray:
        #X_test_scaled = self.prepr_pl.transform(test_X.values)
        X_test_scaled = test_X.values
        preds = self.clf.predict(X_test_scaled)
        return preds

    def reset(self):
        self.clf = DecisionTreeClassifier(random_state=1, **self.clf_args)
        # self.prepr_pl = Pipeline([("min-max", MinMaxScaler())])
        #self.prepr_pl = Pipeline([])
        self.feature_names=[]

    def extract_rules(self):
        tree_rules = export_text(self.clf, feature_names=self.feature_names, decimals=6)
        return tree_rules
