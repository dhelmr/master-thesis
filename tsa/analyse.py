from algorithms.building_block import BuildingBlock
from dataloader.syscall import Syscall
from tsa.preprocessing import Histogram


class TrainingSetAnalyser(BuildingBlock):
    def __init__(self, building_block: BuildingBlock = None):
        super().__init__()
        self._input = building_block
        self._dependency_list = [building_block]
        self._histogram = Histogram()
        self._analyse_result = {}

    def _calculate(self, syscall: Syscall):
        return self._input.get_result(syscall)

    def train_on(self, syscall: Syscall):
        inp = self._input.get_result(syscall)
        self._histogram.add(inp)

    def fit(self):
        self._analyse_result = {
            "unique": len(self._histogram.keys()),
            "total": len(self._histogram)
        }

    def get_analyse_result(self) -> dict:
        return self._analyse_result

    def depends_on(self) -> list:
        return self._dependency_list
