from matplotlib import pyplot as plt
from sklearn.manifold import MDS
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler

from tsa.unsupervised.mixed_model import Histogram
from tsa.unsupervised.preprocessing import OutlierDetector


class ThreadClusteringOD(OutlierDetector):

    def __init__(self, building_block, train_features=None, n_components=2):
        super().__init__(building_block, train_features)
        self._mds = MDS(n_components=n_components, dissimilarity="precomputed")

    def _add_training_data(self, index, ngram, syscall):
        self._training_data.append((index, ngram, syscall.thread_id()))

    def detect_anomalies(self, training_data):
        counts_by_thread = dict()
        for i, ngram, thread_id in training_data:
            if thread_id not in counts_by_thread:
                counts_by_thread[thread_id] = Histogram()
            counts_by_thread[thread_id].add(ngram)

        distance_matrix = []
        for hist1 in counts_by_thread.values():
            row = []
            for hist2 in counts_by_thread.values():
                row.append(hist1.hellinger_distance(hist2))
            distance_matrix.append(row)

        transformed = self._mds.fit_transform(distance_matrix)
        transformed = MinMaxScaler().fit_transform(transformed)
        print(transformed)
        plot(transformed)
        preds = self._do_outlier_detection(transformed)
        anomalous_threads = set()
        for i, thread_id in enumerate(counts_by_thread.keys()):
            if preds[i] < 0:
                anomalous_threads.add(thread_id)

        anomalies = set()
        for i, ngram, thread_id in training_data:
            if thread_id in anomalous_threads:
                anomalies.add(i)
        return anomalies

    def _do_outlier_detection(self, matrix):
        lof = LocalOutlierFactor()
        preds = lof.fit_predict(matrix)
        return preds

def plot(matrix):
    if len(matrix[0]) > 2:
        print("dim > 2, abort")
        return
    X = [x for x, _ in matrix]
    Y = [y for _, y in matrix]
    plt.scatter(X, Y)
    plt.savefig("fig-thread-clustering.png")