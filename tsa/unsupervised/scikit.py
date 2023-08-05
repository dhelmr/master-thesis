from abc import abstractmethod

from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from tsa.unsupervised.preprocessing import OutlierDetector


class ScitkitOD(OutlierDetector):
    def __init__(self, building_block, estimator, train_features=None, only_unique=False, **kwargs):
        super().__init__(building_block, train_features, **kwargs)
        self._only_unique = only_unique
        self._estimator = estimator

    def detect_anomalies(self, training_data):
        if self._only_unique:
            training_data = list(set(training_data))
        else:
            training_data = list(training_data)
        pred = self._estimator.fit_predict(training_data)
        anomalies = set()
        for i, p in enumerate(pred):
            if self._is_anomaly(training_data, i, p):
                anomalies.add(i)
        return anomalies

    def _is_anomaly(self, X, index, pred_value):
        return pred_value < 0


class LOF(ScitkitOD):
    def __init__(self, building_block, train_features=None, only_unique=False, cache_key=None, **kwargs):
        super().__init__(building_block, LocalOutlierFactor(**kwargs), train_features, only_unique, cache_key=cache_key)

class EllipticEnvelopeOD(ScitkitOD):
    def __init__(self, building_block, train_features=None, only_unique=False, cache_key=None, **kwargs):
        super().__init__(building_block, EllipticEnvelope(**kwargs), train_features, only_unique, cache_key=cache_key)

class IsolationForestOD(ScitkitOD):
    def __init__(self, building_block, train_features=None, only_unique=False,cache_key=None, **kwargs):
        super().__init__(building_block, IsolationForest(**kwargs), train_features, only_unique, cache_key=cache_key)

