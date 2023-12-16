import argparse
import os.path
import tempfile

import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from tsa.perf_pred.cv import PerformancePredictor

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from sklearn.tree import export_text
from subprocess import call


class DecisionTree(PerformancePredictor):

    def __init__(self, cli_args=[]):

        parser = argparse.ArgumentParser()
        parser.add_argument("--max-depth", default=5, type=int)
        parser.add_argument("--minmax-scaling", default=False, action="store_true")
        #parser.add_argument("--min-samples-leaf", default=3, type=int)
        args = parser.parse_args(cli_args)
        self.max_depth = args.max_depth
        self.minmax_scaling = args.minmax_scaling

        self.pipeline = None
        self.feature_names = []
        self.reset()

    def train(self, train_X: pandas.DataFrame, train_y: numpy.ndarray):
        train_X = self._preprocess(train_X)
        self.feature_names = list(train_X.columns)
        self.pipeline.fit(train_X.values, train_y)

    def predict(self, test_X: pandas.DataFrame) -> numpy.ndarray:
        test_X = self._preprocess(test_X)
        predictions = self.pipeline.predict(test_X.values)
        return predictions

    def _preprocess(self, df):
        df = df.replace([np.inf, -np.inf], np.nan)
        return df.fillna(0)
    def reset(self):
        self.clf = DecisionTreeClassifier(random_state=1, max_depth=self.max_depth)
        if self.minmax_scaling:
            self.pipeline = Pipeline([("min-max", MinMaxScaler()), ("clf", self.clf)])
        else:
            self.pipeline = Pipeline([("clf", self.clf)])
        self.feature_names=[]

    def extract_rules(self, out_path: str, class_names=None):
        if class_names is None:
            class_names = ["0", "1"]
        tree_rules = export_text(self.clf, feature_names=self.feature_names, decimals=6)
        dt_to_svg(self.clf, feature_names=self.feature_names, target_names=class_names, out_path=out_path)
        return tree_rules

def dt_to_svg(dt, feature_names, target_names, out_path):
    from sklearn.tree import export_graphviz
    # Export as dot file
    tmp_file = tempfile.mktemp(suffix=".dot")
    export_graphviz(dt, out_file=tmp_file,
                    feature_names=feature_names,
                    class_names=target_names, impurity=False,
                    rounded=True, proportion=False,
                    precision=5, filled=True)
    print(tmp_file)

    # Convert to png using system command (requires Graphviz)
    call(['dot', '-Tsvg', tmp_file, '-o', out_path])