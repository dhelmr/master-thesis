from typing import Optional

from dataloader.base_data_loader import BaseDataLoader
from tsa.dataloaders.filter_recording import FilteredRecording


class FilterDataloader(BaseDataLoader):

    def __init__(self, wrapped_dataloader: BaseDataLoader, max_syscalls: Optional[int] = None):
        super().__init__(wrapped_dataloader.scenario_path)
        self.dl = wrapped_dataloader
        self._syscall_training_counter = 0
        self._max_syscalls = max_syscalls
        self._in_training = True

    def training_data(self) -> list:
        return [FilteredRecording(r, self.pass_through) for r in self.dl.training_data()]

    def validation_data(self) -> list:
        self._in_training = False
        return self.dl.validation_data()

    def test_data(self) -> list:
        self._in_training = False
        return self.dl.test_data()

    def extract_recordings(self, category: str) -> list:
        return self.dl.extract_recordings(category)

    def collect_metadata(self) -> dict:
        return self.dl.collect_metadata()

    def get_direction_string(self):
        return self.dl.get_direction_string()

    def cfg_dict(self):
        # TODO implement own superclass for dataloaders
        return self.dl.cfg_dict()

    def pass_through(self, syscall) -> bool:
        if not self._in_training:
            return True
        if self._max_syscalls is None:
            return True
        self._syscall_training_counter += 1
        filter = self._syscall_training_counter <= self._max_syscalls
        return filter
