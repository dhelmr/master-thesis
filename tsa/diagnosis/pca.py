import numpy as np
from sklearn.decomposition import PCA

from algorithms.building_block import BuildingBlock
from dataloader.syscall import Syscall


class PCA_BB(BuildingBlock):
    def __init__(self, input_bb, n_components):
        super().__init__()
        self._input = input_bb
        self._dependency_list = [self._input]
        if n_components is False:
            self.skip = True
        else:
            self.skip = False
            self.pca = PCA(n_components=n_components)
        self._train_data = []

    def train_on(self, syscall: Syscall):
        inp = self._input.get_result(syscall)
        if inp is None:
            return
        self._train_data.append(inp)

    def fit(self):
        self.pca.fit(self._train_data)
        del self._train_data

    def _calculate(self, syscall: Syscall):
        inp = self._input.get_result(syscall)
        if inp is None:
            return None
        #arr = np.array([inp])
        #arr = arr.reshape(1,-1)
        transformed = self.pca.transform([inp])[0]
        return tuple(transformed)

    def depends_on(self) -> list:
        return self._dependency_list
