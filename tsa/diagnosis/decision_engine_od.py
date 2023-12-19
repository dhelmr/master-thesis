from algorithms.building_block import BuildingBlock
from tsa.diagnosis.preprocessing import OutlierDetector


class DecisionEngineOD(OutlierDetector):
    def __init__(self, building_block, decision_engine: BuildingBlock):
        super().__init__(building_block, decision_engine)
        self._de = decision_engine
        if not self._de.is_decider():
            raise ValueError("Expect decision_engine to be a decider.")

    def detect_anomalies(self, training_data):
        for t in training_data:
            self._de._calculate(t)

    def depends_on(self) -> list:
        return super().depends_on() + [self._de]
