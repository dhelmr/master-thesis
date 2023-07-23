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
        self._add_matrix_stats(stats, "ngram_distances", self.ngram_thread_matrix.ngram_distances())
        self._add_matrix_stats(stats, "thread_distances", self.ngram_thread_matrix.thread_distances())
        return [stats]

    def _add_distribution_stats(self, stats, prefix: str, dists, n_classes):
        entropies = []
        for _, dist in dists.items():
            entropies.append(dist.normalized_entropy(n_classes=n_classes))
            self._update_deviation_stats(stats, prefix="%s_norm_entropy" % prefix, items=entropies)

        simpson_indexes = []
        for ngram, dist in dists.items():
            if len(dist.keys()) <= 1:
                # print("skip simpson index, because less or equal 1 classes", ngram)
                continue
            simpson_indexes.append(dist.simpson_index())
        self._update_deviation_stats(stats, prefix="%s-simpson-index" % prefix, items=simpson_indexes)

        self._update_deviation_stats(stats,
                                     prefix="%s-unique" % prefix,
                                     items=[dist.unique_elements() for _, dist in dists.items()])

    def _update_deviation_stats(self, stats, prefix, items):
        stats.update({
            "%s_mean" % prefix: np.mean(items),
            "%s_var" % prefix: np.var(items),
            "%s_min" % prefix: np.min(items),
            "%s_max" % prefix: np.max(items),
            "%s_median" % prefix: np.median(items),
        })

    def _add_matrix_stats(self, stats, prefix, matrix):
        matrix_stats = {
            "mean": np.mean(matrix),
            "var": np.var(matrix),
            "std": np.std(matrix)
        }
        nuc_norm = np.linalg.norm(matrix, ord="nuc")
        matrix_stats["(nuc)-norm"] = nuc_norm
        for norm_ord in ["fro", 1, -1, 2, -2]:
            norm = np.linalg.norm(matrix, ord=norm_ord)
            matrix_stats["(%s)-norm" % norm_ord] = norm
            matrix_stats["(%s)-norm/nuc-norm" % norm_ord] = norm/nuc_norm
        matrix_stats = {"%s-%s" % (prefix, key): v for key, v in matrix_stats.items()}
        stats.update(matrix_stats)
        self._add_eigenval_stats(stats, prefix, matrix)

    def _add_eigenval_stats(self, stats, prefix, matrix):
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
