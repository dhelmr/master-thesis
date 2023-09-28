import matplotlib.pyplot as plt
import numpy
import pandas
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from tsa.perf_pred.cv import PerformancePredictor

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


class DecisionTree(PerformancePredictor):

    def __init__(self):
        self.clf = DecisionTreeClassifier(random_state=1)
        self.prepr_pl = Pipeline([("min-max", MinMaxScaler())])

    def train(self, train_X: pandas.DataFrame, train_y: numpy.ndarray):
        train_set = train_X.values
        X_train_scaled = self.prepr_pl.fit_transform(train_set)
        self.clf.fit(X_train_scaled, train_y)

    def predict(self, test_X: pandas.DataFrame) -> numpy.ndarray:
        X_test_scaled = self.prepr_pl.transform(test_X.values)
        preds = self.clf.predict(X_test_scaled)
        return preds
