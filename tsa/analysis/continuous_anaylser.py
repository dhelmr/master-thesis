import numpy as np

from tsa.analysis.analyser import AnalyserBB


class ContinuousTrainingSetAnalyser(AnalyserBB):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data = []
    def _add_input(self, inp):
        if inp is None:
            return #TODO
        self._data.append(inp)

    def _make_stats(self):
        data = np.array(self._data)
        variance = data.var(axis=0)
        stats = {}
        for i, var in enumerate(variance):
            stats[f"variance_{i}"] = var
        return stats