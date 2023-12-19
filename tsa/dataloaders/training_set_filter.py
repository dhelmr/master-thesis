from typing import Optional

from algorithms.building_block import BuildingBlock, IDSPhase
from dataloader.syscall import Syscall


class TrainingSetFilter(BuildingBlock):
    def __init__(self, input_bb, max_syscalls: Optional[int]):
        super().__init__()
        self._input = input_bb
        self._dependency_list = [self._input]
        self._syscall_training_counter = 0
        self._max_syscalls = max_syscalls

    def _calculate(self, syscall: Syscall):
        if self._ids_phase is not IDSPhase.TRAINING:
            return self._input.get_result(syscall)
        if (
            self._max_syscalls is not None
            and self._syscall_training_counter > self._max_syscalls
        ):
            return None
        self._syscall_training_counter += 1
        return self._input.get_result(syscall)

    def depends_on(self) -> list:
        return self._dependency_list
