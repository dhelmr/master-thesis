"""
Building Block for max value of training threshold.
"""
import numpy as np

from dataloader.syscall import Syscall

from algorithms.building_block import BuildingBlock


class PercentileThreshold(BuildingBlock):
    """
        Saves maximum anomaly score of validation data as threshold.
    """

    def __init__(self,
                 feature: BuildingBlock,
                 percentile=95):
        super().__init__()
        self._percentile = percentile
        self._threshold = None

        self._feature = feature
        self._dependency_list = []
        self._dependency_list.append(self._feature)
        self._anomaly_values = []

    def depends_on(self):
        return self._dependency_list

    def val_on(self, syscall: Syscall):
        """
        save highest seen anomaly_score
        """
        anomaly_score = self._feature.get_result(syscall)
        if isinstance(anomaly_score, (int, float)):
            self._anomaly_values.append(anomaly_score)

    def _calculate(self, syscall: Syscall) -> bool:
        """
        If needed, calculate threshold (defined as n-th percentile of all anomaly values)
        Return 0 if anomaly_score is below threshold.
        Otherwise return 1.
        """
        if self._threshold is None:
            self._threshold = np.percentile(self._anomaly_values, self._percentile)
        anomaly_score = self._feature.get_result(syscall)
        if isinstance(anomaly_score, (int, float)):
            if anomaly_score > self._threshold:
                return True
        return False

    def is_decider(self):
        return True
