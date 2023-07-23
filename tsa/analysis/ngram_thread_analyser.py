import numpy as np
from numpy import e

from tsa.analysis.analyser import AnalyserBB
from tsa.ngram_thread_pca import NgramThreadMatrix


class NgramThreadAnalyser(AnalyserBB):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ngram_thread_matrix = NgramThreadMatrix()

    def _add_input(self, syscall, inp):
        if inp is None:
            # TODO
            return
        ngram = inp
        thread_id = syscall.thread_id()
        self.ngram_thread_matrix.add(ngram, thread_id)

    def _make_stats(self):
        matrix, _, _ = self.ngram_thread_matrix.ngram_thread_matrix()
        unique = len(self.ngram_thread_matrix.distributions_by_ngram().keys())
        total = self.ngram_thread_matrix.total_ngrams
        stats ={
            "unique": unique,
            "total": total,
            "n_threads": len(self.ngram_thread_matrix.distributions_by_threads().keys()),
            "u/t": unique/total
        }
        self._add_distribution_stats(stats, "ngram_dists",  self.ngram_thread_matrix.distributions_by_ngram())
        self._add_distribution_stats(stats, "thread_dists", self.ngram_thread_matrix.distributions_by_threads())

        self._add_matrix_stats(stats, "ngramXthreads", matrix)
        self._add_matrix_stats(stats, "ngram_distances", self.ngram_thread_matrix.ngram_distances())
        self._add_matrix_stats(stats, "thread_distances", self.ngram_thread_matrix.thread_distances())
        return [stats]

    def _add_distribution_stats(self, stats, prefix: str, dists):

        entropies = []
        for _, dist in dists.items():
            entropies.append(dist.entropy())
        stats.update({
            "%s_entropy_mean" % prefix: np.mean(entropies),
            "%s_entropy_var" % prefix: np.var(entropies),
            "%s_entropy_min" % prefix: np.min(entropies),
            "%s_entropy_max" % prefix: np.max(entropies),
            "%s_entropy_median" % prefix: np.median(entropies),
        })

    def _add_matrix_stats(self, stats, prefix, matrix):
        matrix_stats = {
            "mean": np.mean(matrix),
            "var": np.var(matrix),
            "std": np.std(matrix)
        }
        for norm_ord in ["fro", "nuc", 1, -1, 2, -2]:
            matrix_stats["(%s)-norm" % norm_ord] = np.linalg.norm(matrix, ord=norm_ord)
        matrix_stats = {"%s-%s" % (prefix, key): v for key, v in matrix_stats.items()}
        stats.update(matrix_stats)
        self._add_eigenval_stats(stats, prefix, matrix)

    def _add_eigenval_stats(self, stats, prefix, matrix):
        matrix = np.array(matrix)
        cov = np.cov(matrix)
        eigenvals = np.linalg.eigvals(cov)
        sorted_vals = [np.real(v) for v in list(sorted(eigenvals, reverse=True))]
        for i, eigenval in enumerate(sorted_vals[:5]):
            stats["%s_eigenval_%s" % (prefix, i) ] = eigenval
        stats["%s_n_eigenvalues" % prefix] = len(sorted_vals)
        stats["%s_eigenval_mean_5" % prefix] = np.mean(sorted_vals[:5])
        stats["%s_eigenval_mean_10" % prefix] = np.mean(sorted_vals[:10])
        stats["%s_eigenval_mean_20" % prefix] = np.mean(sorted_vals[:20])