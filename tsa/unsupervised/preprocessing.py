import abc
import hashlib
import os
import pickle
from typing import List, Set, Optional

from algorithms.building_block import BuildingBlock, IDSPhase
from dataloader.syscall import Syscall


class OutlierDetector(BuildingBlock):

    def __init__(self, building_block: BuildingBlock = None, train_features=None, cache_key: str = None):
        super().__init__()
        self._input = building_block
        if train_features is None:
            self._train_input = self._input
            self._distinct_train_features = False
            self._dependency_list = [building_block]
        else:
            self._train_input = train_features
            self._distinct_train_features = True
            self._dependency_list = [building_block, train_features]
        self._training_data = list()
        self._anomaly_indexes = list()
        self._fitted = False
        self.anomaly_return_value = None
        self._train_index = -1
        self._test_index = -1
        self._cache_key = cache_key
        if cache_key is not None:
            anomaly_indexes = self._load_anomaly_indexes(cache_key)
            if anomaly_indexes is not None:
                self._anomaly_indexes = anomaly_indexes
                self._fitted = True
                self.train_on = BuildingBlock().train_on
                self.fit = BuildingBlock().fit
                del self._training_data

    def _calculate(self, syscall: Syscall):
        self._test_index += 1
        if not self._fitted:
            raise RuntimeError("Unexpected call of _calculate before fitted. Abort.")
        if self._ids_phase == IDSPhase.TEST:
            # the outlier detector only is running in training mode
            # otherwise, the last building block's output is passed on
            return self._input.get_result(syscall)

        if self._test_index in self._anomaly_indexes:
            return self.anomaly_return_value
        else:
            return self._input.get_result(syscall)

    def train_on(self, syscall: Syscall):
        self._train_index += 1
        ngram = self._train_input.get_result(syscall)
        if ngram is None:
            return
        self._add_training_data(self._train_index, ngram, syscall)

    def _add_training_data(self, index, ngram, syscall):
        self._training_data.append((index, ngram))

    def fit(self):
        self._anomaly_indexes = self.detect_anomalies(self._training_data)
        del self._training_data
        self._fitted = True
        print("Number of Anomalies:", len(self._anomaly_indexes))
        if self._cache_key is not None:
            self._save_anomaly_indexes()

    @abc.abstractmethod
    def detect_anomalies(self, training_data) -> Set[int]:
        raise NotImplemented()

    def depends_on(self) -> list:
        return self._dependency_list

    def _get_cache_path(self, cache_key):
        if "CACHE_PATH" not in os.environ:
            raise KeyError("$CACHE_PATH must be set")
        md5_hash = hashlib.md5(cache_key.encode()).hexdigest()
        path = os.path.join(os.environ["CACHE_PATH"], md5_hash + ".pkl")
        return path

    def _load_anomaly_indexes(self, cache_key) -> Optional[List[int]]:
        path = self._get_cache_path(cache_key)
        if not os.path.exists(path):
            return None
        print("load OD preprocessing result from", path)
        with open(path, "rb") as f:
            return pickle.load(f)

    def _save_anomaly_indexes(self):
        path = self._get_cache_path(self._cache_key)
        if os.path.exists(path):
            return
        print("Write preprocessing result to", path)
        with open(path, "wb") as f:
            pickle.dump(self._anomaly_indexes, f)
