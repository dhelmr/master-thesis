import os
import tempfile
from typing import List

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from numpy.linalg import linalg
from scipy.spatial.distance import squareform, pdist
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler

from tsa.analysis.analyser import AnalyserBB
from tsa.analysis.ngram_thread_matrix import NgramThreadMatrix, process_thread_id
from tsa.diagnosis.thread_clustering import DISTANCE_FN

DEFAULT_DISTANCES = ["euclidean", "cosine", "hamming"]


class ClusterVisualize(AnalyserBB):
    def __init__(self, *args, distances: List[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._matrix = NgramThreadMatrix()
        if distances is None:
            distances = DEFAULT_DISTANCES
        self._distances = distances
        self._stat_index = 0

    def _add_input(self, syscall, inp):
        if inp is None:
            return
        self._matrix.add(inp, process_thread_id(syscall))

    def _make_stats(self):
        nXt, _, _ = self._matrix.ngram_thread_matrix()
        tfidf_nXt, _, _ = self._matrix.tf_idf_matrix()
        tXn = np.transpose(nXt)
        tfidf_tXn = np.transpose(tfidf_nXt)

        stats = []
        self.make_matrix_plots(nXt, stats, "nXt")

        self.make_matrix_plots(tfidf_nXt, stats, "nXt-tfidf")
        self.make_matrix_plots(self.reduce_dim(nXt, 10), stats, "nXt-tfidf-pca10")
        self.make_matrix_plots(self.reduce_dim(nXt, 10), stats, "nXt-pca10")
        self.make_matrix_plots(tXn, stats, "tXn")
        self.make_matrix_plots(tfidf_tXn, stats, "tXn-tfidf")
        self._stat_index += 1
        return stats

    def make_matrix_plots(self, matrix, stats, name):
        for dist in self._distances:
            self._plot_and_stats(matrix, stats, name, dist)
            self._plot_and_stats(self.normalize(matrix), stats, f"{name}-norm", dist)

    def _plot_and_stats(self, matrix, stats, name, distance):
        try:
            coords = self.build_clusters(matrix, distance)
            self.plot(coords, f"{name}-{distance}")
            rows_common = {
                "distance": distance,
                "name": name,
                "mean": np.mean(coords),
                "var": np.var(coords),
            }
            print(rows_common)
            stats.append(rows_common)
        except ValueError as e:
            print("Error occured for", name, distance)
            print(e)
        # for r, c in enumerate(coords):
        #    stats.append({
        #        **{f"x{i}": c[i] for i in range(len(coords[0]))},
        #        **{"i": r},
        #        **rows_common
        #    })
        # coords_df = pandas.DataFrame(coords, columns=[f"coord_{i}" for i in range(len(coords[0]))])
        # log_pandas_df(coords_df, name)

    def reduce_dim(self, matrix, n_components=10):
        pca = PCA(n_components)
        transformed = pca.fit_transform(matrix)
        return MinMaxScaler().fit_transform(transformed)

    def build_clusters(self, matrix, distance):
        if distance in DISTANCE_FN:
            distance = DISTANCE_FN[distance]
        distance_matrix = squareform(pdist(matrix, metric=distance))

        mds = MDS(n_components=2, metric=True, dissimilarity="precomputed")
        # scaled = MinMaxScaler().fit_transform(distance_matrix)
        transformed = mds.fit_transform(distance_matrix)
        scaled = MinMaxScaler().fit_transform(transformed)
        return scaled

    def normalize(self, matrix):
        norm = linalg.norm(matrix, axis=1, ord=1).reshape(-1, 1)
        return matrix / norm

    def plot(self, matrix, name: str):
        if len(matrix[0]) > 2:
            print("dim > 2, transform with pca")
            pca = PCA(n_components=2)
            matrix = pca.fit_transform(matrix)
        X = [x for x, _ in matrix]
        Y = [y for _, y in matrix]
        plt.scatter(X, Y)
        log_plot_to_mlflow(f"{self._stat_index}-{name}")

    # def get_stats(self):
    #    return
    # TODO!


def log_plot_to_mlflow(name):
    tmpfile = tempfile.mkdtemp()
    outpath = os.path.join(tmpfile, f"{name}.png")
    plt.gcf().set_size_inches(21, 14)
    plt.savefig(outpath, dpi=50)
    plt.clf()
    mlflow.log_artifact(outpath)
