import numpy as np

from algorithms.building_block import BuildingBlock
from dataloader.syscall import Syscall

class TupleBB(BuildingBlock):
    def __init__(self, input_bb):
        super().__init__()
        self._input = input_bb
        self._dependency_list = [self._input]

    def depends_on(self) -> list:
        return self._dependency_list

    def _calculate(self, syscall: Syscall):
        inp = self._input.get_result(syscall)
        if inp is None:
            return None
        return tuple([inp])
class W2VConcat(BuildingBlock):
    def __init__(self, input_bb, variance: bool, cosine_sims: bool):
        super().__init__()
        self._input = input_bb
        self._dependency_list = [self._input]
        self.variance = variance
        self.cosine_sims = cosine_sims
    def depends_on(self) -> list:
        return self._dependency_list

    def _calculate(self, syscall: Syscall):
        ngram = self._input.get_result(syscall)
        if ngram is None:
            return None
        arrays = [np.array(el) for el in ngram]
        w2v_mean = np.mean(arrays, axis=0)
        features = w2v_mean.tolist()
        if self.variance:
            variance = np.var(arrays, axis=0)
            features += variance.tolist()
        if self.cosine_sims:
            cosines = []
            for i in range(len(arrays)-1):
                # cosine = np.cos()
                cosine = cos_sim(arrays[i], arrays[i+1])
                cosines.append(cosine)
            features += cosines
        return tuple(features)

def cos_sim(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))