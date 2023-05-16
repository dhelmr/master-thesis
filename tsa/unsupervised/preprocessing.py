import abc

from algorithms.building_block import BuildingBlock, IDSPhase
from dataloader.syscall import Syscall


class OutlierDetector(BuildingBlock):

    def __init__(self, building_block: BuildingBlock = None, train_features=None):
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
        self._anomalies = set()
        self._fitted = False
        self.anomaly_return_value = None

    def _calculate(self, syscall: Syscall):
        if not self._fitted:
            raise RuntimeError("Unexpected call of _calculate before fitted. Abort.")
        if self._ids_phase == IDSPhase.TEST:
            # the outlier detector only is running in training mode
            # otherwise, the last building block's output is passed on
            return self._input.get_result(syscall)

        ngram = self._train_input.get_result(syscall)
        if ngram in self._anomalies:
            return self.anomaly_return_value
        elif self._distinct_train_features:
            return self._input.get_result(syscall)
        else:
            return ngram


    def train_on(self, syscall: Syscall):
        ngram = self._train_input.get_result(syscall)
        if ngram is None:
            return
        self._training_data.append(ngram)

    def fit(self):
        self._anomalies = self.detect_anomalies(self._training_data)
        del self._training_data
        self._fitted = True
        print("Number of Anomalies:", len(self._anomalies))

    @abc.abstractmethod
    def detect_anomalies(self, training_data):
        raise NotImplemented()

    def depends_on(self) -> list:
        return self._dependency_list




