import hashlib
import math
import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import linalg
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import MDS
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler

from scipy.spatial.distance import pdist, cosine, squareform, euclidean

from algorithms.building_block import BuildingBlock
from tsa.ngram_thread_matrix import NgramThreadMatrix, process_thread_id
from tsa.unsupervised.preprocessing import OutlierDetector

OD_METHODS = {
    cls.__name__: cls for cls in [LocalOutlierFactor, IsolationForest, EllipticEnvelope]
}


def binary_jaccard_distance(u, v):
    sum_and = 0
    sum_or = 0
    for i in range(len(u)):
        if u[i] > 0 and v[i] > 0:
            sum_and += 1
        if u[i] > 0 or v[i] > 0:
            sum_or += 1
    return (1 - sum_and / sum_or) if sum_or != 0 else 0

def binary_hamming(u, v):
    distance = 0
    for i in range(len(u)):
        if u[i] > 0 and v[i] > 0:
            continue
        if u[i] == 0 and v[i] == 0:
            continue
        distance += 1
    return distance

_SQRT2 = np.sqrt(2)


def hellinger(u, v):
    return euclidean(np.sqrt(u), np.sqrt(v)) / _SQRT2


def jaccard_hellinger(u, v):
    return binary_jaccard_distance(u, v) * hellinger(u, v)

def jaccard_cosine(u,v ):
    return binary_jaccard_distance(u, v) * cosine(u, v)

DISTANCE_FN = {
    "jaccard-cosine": jaccard_cosine,
    "binary-jaccard": binary_jaccard_distance,
    "hellinger": hellinger,
    "binary-hamming": binary_hamming,
    "jaccard-hellinger": jaccard_hellinger
}


class ThreadClusteringOD(OutlierDetector):

    def __init__(self, building_block, train_features=None, n_components=2, distance="jaccard-cosine", tf_idf=False,
                 skip_mds: bool = False, thread_based=True, normalize_rows=False, normalize_ord=1, metric_mds=True,
                 plot_mds=False, od_method: str = "IsolationForest", fill_na=False, od_kwargs=None, cache_key=None, **kwargs):
        super().__init__(building_block, train_features, cache_key)
        if od_kwargs is None:
            od_kwargs = {}
        self._mds = MDS(n_components=n_components, metric=metric_mds, dissimilarity="precomputed")
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
        self._thread_based = thread_based
        self._normalize_rows = normalize_rows
        self._normalize_ord = normalize_ord
        self._fill_na = fill_na
        self._plot_mds = plot_mds
        self._cache_key = cache_key
        self._skip_training_data = self._cache_key is not None and os.path.exists(self._cache_key_to_path(self._cache_key))
        self._matrix_builder = None if self._skip_training_data else NgramThreadMatrix()
        if self._skip_training_data:
            self.train_on = BuildingBlock().train_on
    def _add_training_data(self, index, ngram, syscall):
        if self._skip_training_data:
            return
        # avoid possibility of ambiguity by using process id+thread_id
        thread_id = process_thread_id(syscall)
        self._training_data.append((index, ngram, thread_id))
        self._matrix_builder.add(ngram, thread_id)

    def _cache_key_to_path(self, cache_key: str):
        if "IDS_CACHE_PATH" not in os.environ:
            raise KeyError("$W2V_CACHE_PATH must be set")

        md5_hash = hashlib.md5(cache_key.encode()).hexdigest()
        model_path = os.path.join(os.environ["CACHE_PATH"], "%s.ngram-matrix.pickle" % md5_hash)
        return model_path

    def _load_data_from_cache(self):
        path = self._cache_key_to_path(self._cache_key)
        if not os.path.exists(path):
            return None, None
        with open(path, "rb") as f:
            print("Open cache: %s" % path)
            data = pickle.load(f)
            return data["matrix"], data["training_data"]

    def _store_cache(self, training_data, matrix):
        path = self._cache_key_to_path(self._cache_key)
        if os.path.exists(path):
            print("cache %s exists, skip" % path)
            return
        with open(path, "wb") as f:
            print("Write to cache: %s" % path)
            pickle.dump({
                "matrix": matrix, "training_data": training_data
            }, f)
    def detect_anomalies(self, training_data):
        # counts_by_thread = dict()
        #
        #    if thread_id not in counts_by_thread:
        #        counts_by_thread[thread_id] = Histogram()
        #    counts_by_thread[thread_id].add(ngram)
        print("Prepare anomaly detection...")
        if self._cache_key is not None:
            matrix_cache, training_data_cache = self._load_data_from_cache()
            if training_data_cache is not None:
                print("Load training data from cache")
                training_data = training_data_cache
            if matrix_cache is not None:
                print("Load ngram-thread matrix from cache")
                self._matrix_builder = matrix_cache

        if self._cache_key is not None:
            self._store_cache(training_data, self._matrix_builder)

        print("number of threads in training set: ", len(self._matrix_builder.threads()))
        print("number of ngrams in training set: ", len(self._matrix_builder.ngrams()))
        # distance_matrix = make_distance_matrix(counts_by_thread, self._distance)
        if self._tf_idf:
            print("Calculate tf_idf matrix")
            matrix, ngrams, threads = self._matrix_builder.tf_idf_matrix()
        else:
            print("Calculate ngram-thread matrix matrix")
            matrix, ngrams, threads = self._matrix_builder.ngram_thread_matrix()
        if self._thread_based:
            print("Transpose matrix...")
            matrix = np.transpose(matrix)  # convert ngram-thread matrix to thread-ngram matrix
        if self._normalize_rows:
            print("Normalize matrix...")
            norm = linalg.norm(matrix, axis=1, ord=self._normalize_ord).reshape(-1, 1)
            matrix = matrix / norm
        print("Calculate distance matrix...")
        distance_matrix = squareform(pdist(matrix, metric=self._distance))
        if self._fill_na:
            distance_matrix = np.where(np.isnan(distance_matrix), 1, distance_matrix)
            distance_matrix = np.where(np.isinf(distance_matrix), 1, distance_matrix)


        preds = self._do_outlier_detection(distance_matrix)
        anomalous_entities = set()
        if self._thread_based:
            for i, thread_id in enumerate(threads):
                if preds[i] < 0:
                    anomalous_entities.add(thread_id)
        else:
            for i, ngram in enumerate(ngrams):
                if preds[i] < 0:
                    anomalous_entities.add(ngram)

        anomalies = set()
        for i, ngram, thread_id in training_data:
            if (self._thread_based and thread_id in anomalous_entities) \
                    or (not self._thread_based and ngram in anomalous_entities):
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
            if self._plot_mds:
                plot(X)
        print("Start Outlier Detection with", self._od.__dict__)
        preds = self._od.fit_predict(X)
        return preds



def plot(matrix):
    if len(matrix[0]) > 2:
        print("dim > 2, transform with pca")
        pca = PCA(n_components=2)
        matrix = pca.fit_transform(matrix)
    X = [x for x, _ in matrix]
    Y = [y for _, y in matrix]
    plt.scatter(X, Y)
    plt.savefig("fig-thread-clustering.png")
