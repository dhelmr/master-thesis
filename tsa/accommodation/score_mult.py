from typing import List

from algorithms.building_block import BuildingBlock


class ScoreMultiplication(BuildingBlock):
    """
    Multiplies the scores of multiple building blocks
    """

    def __init__(self, input: List[BuildingBlock]):
        super().__init__()
        # parameter
        self._input = input
        # dependency list
        self._dependency_list = input
        print("input:", self._input)

    def depends_on(self):
        return self._dependency_list

    def _calculate(self, syscall):
        product = 1
        for bb in self._input:
            score = bb.get_result(syscall)
            if score is None:
                return None
            product *= score
        return product
