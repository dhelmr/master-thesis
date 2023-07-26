import math

import numpy as np
from matplotlib import pyplot as plt
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.manifold import MDS
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler

from scipy.spatial.distance import pdist, cosine, squareform
from tsa.ngram_thread_matrix import NgramThreadMatrix
from tsa.unsupervised.preprocessing import OutlierDetector

OD_METHODS = {
    cls.__name__: cls for cls in [LocalOutlierFactor, IsolationForest, EllipticEnvelope]
}

def binary_jaccard_distance(u,v):
    sum_and = 0
    sum_or = 0
    for i in range(len(u)):
        if u[i] > 0 and v[i] > 0:
            sum_and += 1
        if u[i] > 0 or v[i] > 0:
            sum_or += 1
    return (1-sum_and/sum_or) if sum_or != 0 else 0

DISTANCE_FN = {
    "jaccard-cosine": lambda u,v: binary_jaccard_distance(u,v)*cosine(u,v),
    "binary-jaccard": binary_jaccard_distance
    # TODO hellinger, ...
}

class ThreadClusteringOD(OutlierDetector):

    def __init__(self, building_block, train_features=None, n_components=2, distance="jaccard-cosine", tf_idf=False,
                 skip_mds: bool = False, od_method: str = "IsolationForest", od_kwargs=None):
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
        if distance in DISTANCE_FN:
            self._distance = DISTANCE_FN[distance]
        else:
            # TODO: check if distance is valid scipy distance fn
            self._distance = distance
        self._tf_idf = tf_idf

    def _add_training_data(self, index, ngram, syscall):
        self._training_data.append((index, ngram, syscall.thread_id()))

    def detect_anomalies(self, training_data):
        #counts_by_thread = dict()
        #
        #    if thread_id not in counts_by_thread:
        #        counts_by_thread[thread_id] = Histogram()
        #    counts_by_thread[thread_id].add(ngram)
        matrix_builder = NgramThreadMatrix()
        for i, ngram, thread_id in training_data:
            matrix_builder.add(ngram, thread_id)


        print("number of threads in training set: ", len(matrix_builder.threads()))
        print("number of ngrams in training set: ", len(matrix_builder.ngrams()))
                # distance_matrix = make_distance_matrix(counts_by_thread, self._distance)
        if self._tf_idf:
            print("Calculate tf_idf matrix")
            matrix, ngrams, threads = matrix_builder.tf_idf_matrix()
        else:
            matrix, ngrams, threads = matrix_builder.ngram_thread_matrix()
        matrix = np.transpose(matrix) # convert ngram-thread matrix to thread-ngram matrix
        print("Calculate distance matrix...")
        print(matrix)
        # TOD support jaccard-weighted distances
        distance_matrix = squareform(pdist(matrix, metric=self._distance))

        preds = self._do_outlier_detection(distance_matrix)
        anomalous_threads = set()
        for i, thread_id in enumerate(threads):
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