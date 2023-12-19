import os
import tempfile

import matplotlib.pyplot as plt
import mlflow
import numpy as np

from tsa.analysis.analyser import AnalyserBB

DEFAULT_DISTANCES = ["euclidean", "cosine", "hamming"]


class FrequencyDistribution(AnalyserBB):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._frequencies = dict()

    def _add_input(self, syscall, inp):
        if inp is None:
            return
        if inp not in self._frequencies:
            self._frequencies[inp] = 0
        self._frequencies[inp] += 1

    def _make_stats(self):
        frequencies = sorted(self._frequencies.values(), reverse=True)

        # threshold = sum(frequencies) * 0.001
        # print(frequencies, threshold)
        ax = self._plot(frequencies)
        # ax.hlines(threshold, 0, len(frequencies), colors="red")
        log_plot_to_mlflow("freq_distr-all")

        # cleaned_freq = [f for f in frequencies if f > threshold]
        # ax = self._plot(cleaned_freq)
        # ax.set(xlim=(0, len(frequencies)))
        # log_plot_to_mlflow("freq_distr_cleaned")

    def _plot(self, frequencies):
        n_freq = len(frequencies)
        fig, ax = plt.subplots()
        x = 0.5 + np.arange(n_freq)
        ax.bar(x, frequencies, width=1, edgecolor="white", linewidth=0.8)

        ax.set(
            xlim=(0, n_freq),
            xlabel="n-gram frequency rank",
            ylabel="frequency",
            ylim=(0, frequencies[0]),
        )
        return ax


def log_plot_to_mlflow(name):
    tmpfile = tempfile.mkdtemp()
    outpath = os.path.join(tmpfile, f"{name}.png")
    plt.gcf().set_size_inches(21, 14)
    plt.savefig(outpath, dpi=50)
    plt.clf()
    mlflow.log_artifact(outpath)
