from matplotlib import pyplot as plt
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.manifold import MDS
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler

from tsa.unsupervised.mixed_model import Histogram
from tsa.unsupervised.preprocessing import OutlierDetector

OD_METHODS = {
    cls.__name__: cls for cls in [LocalOutlierFactor, IsolationForest, EllipticEnvelope]
}
class ThreadClusteringOD(OutlierDetector):

    def __init__(self, building_block, train_features=None, n_components=2, skip_mds: bool = False,
                 od_method: str = "IsolationForest", od_kwargs=None):
        super().__init__(building_block, train_features)
        if od_kwargs is None:
            od_kwargs = {}
        self._mds = MDS(n_components=n_components, dissimilarity="precomputed")
        if od_method not in OD_METHODS:
            raise ValueError("Invalid od_method name %s. Must be one of: %s" % (od_method, list(OD_METHODS.keys())))
        if od_method == "LocalOutlierFactor" and skip_mds:
            od_kwargs["metric"] = "precomputed"
            self._need_mds = False
        elif od_method != "LocalOutlierFactor" and skip_mds:
            raise ValueError("skip_mds cannot be set to True for %s" % od_method)
        else:
            self._need_mds = True
        self._od = OD_METHODS[od_method](**od_kwargs)


    def _add_training_data(self, index, ngram, syscall):
        self._training_data.append((index, ngram, syscall.thread_id()))

    def detect_anomalies(self, training_data):
        counts_by_thread = dict()
        for i, ngram, thread_id in training_data:
            if thread_id not in counts_by_thread:
                counts_by_thread[thread_id] = Histogram()
            counts_by_thread[thread_id].add(ngram)

        print("number of threads in training set: ", len(counts_by_thread))
        print("Calculate distance matrix...")
        distance_matrix = []
        for hist1 in counts_by_thread.values():
            row = []
            for hist2 in counts_by_thread.values():
                row.append(hist1.hellinger_distance(hist2))
            distance_matrix.append(row)

        preds = self._do_outlier_detection(distance_matrix)
        anomalous_threads = set()
        for i, thread_id in enumerate(counts_by_thread.keys()):
            if preds[i] < 0:
                anomalous_threads.add(thread_id)

        anomalies = set()
        for i, ngram, thread_id in training_data:
            if thread_id in anomalous_threads:
                anomalies.add(i)
        return anomalies

    def _do_outlier_detection(self, distance_matrix):
        X = distance_matrix
        if self._need_mds:
            print("Start MDS...")
            transformed = self._mds.fit_transform(distance_matrix)
            transformed = MinMaxScaler().fit_transform(transformed)
            print("finished MDS")
            X = transformed
            plot(X)
        print("Start Outlier Detection with", self._od.__dict__)
        preds = self._od.fit_predict(X)
        return preds
def plot(matrix):
    if len(matrix[0]) > 2:
        print("dim > 2, abort")
        return
    X = [x for x, _ in matrix]
    Y = [y for _, y in matrix]
    plt.scatter(X, Y)
    plt.savefig("fig-thread-clustering.png")