from sklearn.neighbors import LocalOutlierFactor

from tsa.unsupervised.preprocessing import OutlierDetector


class LOF(OutlierDetector):
    def __init__(self, building_block, train_features=None, only_unique=False, **kwargs):
        super().__init__(building_block, train_features)
        self.lof = LocalOutlierFactor(**kwargs)
        self._only_unique = only_unique
    def detect_anomalies(self, training_data):
        if self._only_unique:
            training_data = list(set(training_data))
        else:
            training_data = list(training_data)
        pred = self.lof.fit_predict(training_data)
        print(pred)
        anomalies = set()
        for i, p in enumerate(pred):
            if p < 0:
                anomalies.add(training_data[i])
        print("%s anomalies found", len(anomalies))
        return anomalies
