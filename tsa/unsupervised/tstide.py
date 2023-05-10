from tsa.unsupervised.preprocessing import OutlierDetector


class TStide(OutlierDetector):

    def __init__(self, building_block, train_features=None, threshold = 3):
        super().__init__(building_block, train_features)
        self._threshold = threshold
    def detect_anomalies(self, training_data):
        counts = {}
        for t in training_data:
            if t not in counts:
                counts[t] = 0
            counts[t] += 1
        anomalies = set()
        for t, count in counts.items():
            if count <= self._threshold:
                anomalies.add(t)
        return anomalies
