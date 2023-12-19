import math
import typing

import pandas as pd

from algorithms.building_block import BuildingBlock
from tsa.analysis.analyser import AnalyserBB
from tsa.histogram import Histogram


class DataDriftAnalyser(AnalyserBB):
    def __init__(self, input_bb: BuildingBlock, update_interval=1000, fixed_stops=None):
        super().__init__(input_bb, update_interval, fixed_stops, test_phase=True)
        self._train_histogram = Histogram()
        self._training_histograms_by_syscalls: typing.Dict[
            int, Histogram
        ] = {}  # stores copies of the histograms at a certain syscall value
        self._test_histogram = Histogram()

    def _add_input(self, syscall, inp):
        if inp is None:
            return
        self._train_histogram.add(inp)

    def _add_test_input(self, syscall, inp):
        self._test_histogram.add(inp)

    def _make_stats(self) -> typing.Union[typing.List[dict], dict]:
        # copy current training set histogram; the stats are calculated later once the test set is available
        # see get_stats()
        self._training_histograms_by_syscalls[
            self._current_i
        ] = self._train_histogram.copy()
        return None

    def get_stats(self):
        print("#ngrams in test set: %s" % self._test_histogram.unique_elements())
        print(
            "#ngrams in complete train set: %s"
            % self._train_histogram.unique_elements()
        )
        stats = []
        max_syscalls = max(self._training_histograms_by_syscalls.keys())
        for syscalls, train_hist in self._training_histograms_by_syscalls.items():
            row = {
                "syscalls": syscalls,
                "is_last": max_syscalls == syscalls,
                **self._calc_data_drift_measures(train_hist, self._test_histogram),
            }
            stats.append(row)
        return pd.DataFrame(stats)

    def _calc_data_drift_measures(self, train_hist, test_hist):
        jsd = train_hist.jensen_shannon_divergence(test_hist)

        train_ngrams = train_hist.keys()
        unseen_test_ngrams = 0
        unseen_unique_test_ngrams = 0
        for test_ngram, count in test_hist:
            if test_ngram not in train_ngrams:
                unseen_test_ngrams += count
                unseen_unique_test_ngrams += 1

        return {
            "jensen_shannon_divergence": jsd,
            "jensen_shannon_distance": math.sqrt(jsd),
            "ratio_unseen_test_ngrams": unseen_test_ngrams / len(test_hist),
            "ratio_unseen_unique_test_ngrams": unseen_unique_test_ngrams
            / test_hist.unique_elements(),
        }
