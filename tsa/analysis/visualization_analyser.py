import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from tsa.analysis.analyser import AnalyserBB


class Visualize(AnalyserBB):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data = []
        self._fig_index = 0
    def _add_input(self, syscall, inp):
        if inp is None:
            return # TODO?
        self._data.append(inp)

    def _make_stats(self):
        #df = pandas.DataFrame(self._data)
        #df.plot(x=df[0], y=df[1])
        pca = PCA(n_components=2)
        data = np.array(self._data)
        reduced = pca.fit_transform(data)
        X = [x for x,_ in reduced]
        Y = [y for _,y in reduced]
        print("explained var", pca.explained_variance_)
        print("explained var-ratio", pca.explained_variance_ratio_)
        plt.scatter(X,Y)
        plt.savefig("fig-%s" % self._fig_index)
        self._fig_index += 1

    def get_stats(self):
        return