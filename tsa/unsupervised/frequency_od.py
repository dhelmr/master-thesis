from tsa.unsupervised.preprocessing import OutlierDetector


class FrequencyOD(OutlierDetector):

    def __init__(self, building_block, train_features=None, threshold=3, relative=False):
        super().__init__(building_block, train_features)
        self._relative = relative
        self._threshold = threshold

    def _get_threshold(self, training_data):
        if self._relative:
            len_td = len(training_data)
            t = self._threshold * len_td
            print(f"Set threshold to {self._threshold} of training data (size={len_td}) => t={t}")
            return t
        else:
            return self._threshold

    def detect_anomalies(self, training_data):
        threshold = self._get_threshold(training_data)
        print(f"Remove all ngrams with frequency < {threshold}")
        counts = {}
        for i, t in training_data:
            if t not in counts:
                counts[t] = 0
            counts[t] += 1
        anomalies = set()
        for i, t in training_data:
            if counts[t] <= threshold:
                anomalies.add(i)
        return anomalies
