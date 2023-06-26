from tsa.unsupervised.preprocessing import OutlierDetector


class FrequencyOD(OutlierDetector):

    def __init__(self, building_block, train_features=None, threshold = 3):
        super().__init__(building_block, train_features)
        self._threshold = threshold

    def detect_anomalies(self, training_data):
        counts = {}
        for i, t in training_data:
            if t not in counts:
                counts[t] = 0
            counts[t] += 1
        anomalies = set()
        for i, t in training_data:
            if counts[t] <= self._threshold:
                anomalies.add(i)
        return anomalies
