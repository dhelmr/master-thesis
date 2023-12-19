import math

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from tsa.analysis.analyser import AnalyserBB
from tsa.analysis.ngram_thread_matrix import NgramThreadMatrix, process_thread_id


class NgramThreadAnalyser(AnalyserBB):
    def __init__(self, *args, eigenvalues=False, matrix_norms=False, pdists=False, pca=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.ngram_thread_matrix = NgramThreadMatrix()
        self.eigenvalues = eigenvalues
        self.matrix_norms = matrix_norms
        self.pdists = pdists
        self.pca = pca

    def _add_input(self, syscall, inp):
        if inp is None:
            # TODO
            return
        ngram = inp
        self.ngram_thread_matrix.add(ngram, process_thread_id(syscall))

    def _make_stats(self):
        matrix, _, _ = self.ngram_thread_matrix.ngram_thread_matrix()
        unique_ngrams = len(self.ngram_thread_matrix.distributions_by_ngram().keys())
        total = self.ngram_thread_matrix.total_ngrams
        n_threads = len(self.ngram_thread_matrix.distributions_by_threads().keys())
        stats = {
            "unique_ngrams": unique_ngrams,
            "total": total,
            "n_threads": n_threads,
            "unique_ngrams/total": unique_ngrams / total,
            "unique_ngrams/n_threads": unique_ngrams / n_threads
        }
        self._add_distribution_stats(stats, "ngram_dists",
                                     self.ngram_thread_matrix.distributions_by_ngram(),
                                     n_classes=n_threads)
        self._add_distribution_stats(stats, "thread_dists",
                                     self.ngram_thread_matrix.distributions_by_threads(),
                                     n_classes=unique_ngrams)

        self._add_matrix_stats(stats, "ngramXthreads", matrix)
        if self.pdists:
            self._add_pdist_stats(stats, "ngram_dists", matrix)
            self._add_pdist_stats(stats, "thread_dists", np.transpose(matrix))
        if self.pca:
            self._add_pca_stats(stats, "nXt_pca", matrix)
            self._add_pca_stats(stats, "tXn_pca",  np.transpose(matrix))
        return [stats]

    def _add_distribution_stats(self, stats, prefix: str, dists, n_classes):
        entropies = []
        for _, dist in dists.items():
            entropies.append(dist.normalized_entropy(n_classes=n_classes))
            self._update_deviation_stats(stats, prefix="%s_norm_entropy" % prefix, items=entropies)

        simpson_indexes = []
        gini_coeffs = []
        zipf_params_a = []
        zipf_params_loc = []
        for ngram, dist in dists.items():
            if len(dist.keys()) <= 1:
                # print("skip simpson index, because less or equal 1 classes", ngram)
                continue
            simpson_indexes.append(dist.simpson_index())
            gini_coeffs.append(dist.gini_coeff())
            zipf = dist.fit_zipf()
            zipf_params_a.append(zipf["a"])
            zipf_params_loc.append(zipf["loc"])
        self._update_deviation_stats(stats, prefix="%s-simpson-index" % prefix, items=simpson_indexes)
        self._update_deviation_stats(stats, prefix="%s-gini-coeff" % prefix, items=gini_coeffs)
        self._update_deviation_stats(stats, prefix="%s-zipf_a" % prefix, items=zipf_params_a)
        self._update_deviation_stats(stats, prefix="%s-zipf_loc" % prefix, items=zipf_params_loc)
        self._update_deviation_stats(stats, prefix="%s-simpson-index" % prefix, items=simpson_indexes)

        self._update_deviation_stats(stats,
                                     prefix="%s-unique" % prefix,
                                     items=[dist.unique_elements() for _, dist in dists.items()])

    def _update_deviation_stats(self, stats, prefix, items):
        v = np.var(items)
        m = np.mean(items)
        stats.update({
            "%s_mean" % prefix: m,
            "%s_var" % prefix: v,
            "%s_min" % prefix: np.min(items) if len(items) > 0 else math.nan,
            "%s_max" % prefix: np.max(items) if len(items) > 0 else math.nan,
            "%s_median" % prefix: np.median(items) if len(items) > 0 else math.nan,
            "%s_iod" % prefix: v / m
        })

    def _add_matrix_stats(self, stats, prefix, matrix):
        self._update_deviation_stats(stats, prefix, matrix)
        matrix_stats = {}
        if self.matrix_norms:
            nuc_norm = np.linalg.norm(matrix, ord="nuc")
            matrix_stats["(nuc)-norm"] = nuc_norm
            for norm_ord in ["fro", 1, -1, 2, -2]:
                norm = np.linalg.norm(matrix, ord=norm_ord)
                matrix_stats["(%s)-norm" % norm_ord] = norm
                matrix_stats["(%s)-norm/nuc-norm" % norm_ord] = norm / nuc_norm
        matrix_stats = {"%s-%s" % (prefix, key): v for key, v in matrix_stats.items()}
        stats.update(matrix_stats)
        self._add_eigenval_stats(stats, prefix, matrix)

    def _add_eigenval_stats(self, stats, prefix, matrix):
        if not self.eigenvalues:
            return
        matrix = np.array(matrix)
        cov = np.cov(matrix)
        eigenvals = np.linalg.eigvals(cov)
        sorted_vals = [np.real(v) for v in list(sorted(eigenvals, reverse=True))]
        for i, eigenval in enumerate(sorted_vals[:5]):
            stats["%s_eigenval_%s" % (prefix, i)] = eigenval
        stats["%s_n_eigenvalues" % prefix] = len(sorted_vals)
        stats["%s_eigenval_mean_5" % prefix] = np.mean(sorted_vals[:5])
        stats["%s_eigenval_mean_10" % prefix] = np.mean(sorted_vals[:10])
        stats["%s_eigenval_mean_20" % prefix] = np.mean(sorted_vals[:20])

    def _add_pdist_stats(self, stats, prefix, matrix):
        for metric in ["jensenshannon", "jaccard", "correlation", "cosine", "canberra", "euclidean", "braycurtis"]:
            distance_matrix = pdist(matrix, metric)
            self._add_matrix_stats(stats, prefix=f"{prefix}-{metric}", matrix=distance_matrix)

    def _add_pca_stats(self, stats, prefix, matrix):
        pca = PCA(n_components=10)
        minmax = MinMaxScaler()
        scaled = minmax.fit_transform(matrix)
        pca.fit_transform(scaled)

        pca_stats = {"pca-noise-var": pca.noise_variance_,}
        for i, evr in enumerate(pca.explained_variance_ratio_):
            pca_stats[f"pca-evr-n{i}"] = evr
        for i, ev in enumerate(pca.explained_variance_):
            pca_stats[f"pca-ev-n{i}"] = ev
        pca_stats["pca-ev-sum"] = sum(pca.explained_variance_)
        pca_stats["pca-evr-sum"] = sum(pca.explained_variance_ratio_)

        stats.update({"%s-%s" % (prefix, k) : v for k,v in pca_stats.items()})