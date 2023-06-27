from copy import deepcopy
from typing import Optional, List

from tsa.dataloaders.filter_recording import FilteredRecording
from tsa.dataloaders.syscall_filter import MaxSyscallFilter
from tsa.dataloaders.tsa_base_dl import TsaBaseDataloader


class FilterDataloader(TsaBaseDataloader):

    def __init__(self, wrapped_dataloader: TsaBaseDataloader, max_syscalls: Optional[int] = None):
        super().__init__(wrapped_dataloader.scenario_path)
        self._validation_ratio = 0.2 # TODO: load dynamically from base dataloader # wrapped_dataloader.get_val_ratio() # TODO abstract Dataloader class
        self.dl = wrapped_dataloader
        self._max_syscalls = max_syscalls
        self._max_syscalls_training = self._max_syscalls
        self._max_syscalls_validation = int(self._max_syscalls * self._validation_ratio / (1-self._validation_ratio)) if self._max_syscalls is not None else None
        self._applied_training_filters = []
        self._applied_val_filters = []

    def training_data(self) -> list:
        f = MaxSyscallFilter(self._max_syscalls_training)
        self._applied_training_filters.append(f)
        return [FilteredRecording(r, f) for r in self.dl.training_data()]

    def validation_data(self) -> list:
        f = MaxSyscallFilter(self._max_syscalls_validation)
        self._applied_val_filters.append(f)
        return [FilteredRecording(r, f) for r in self.dl.validation_data()]

    def test_data(self) -> list:
        return self.dl.test_data()

    def extract_recordings(self, category: str) -> list:
        return self.dl.extract_recordings(category)

    def collect_metadata(self) -> dict:
        return self.dl.collect_metadata()

    def get_direction_string(self):
        return self.dl.get_direction_string()

    def cfg_dict(self):
        # TODO implement own superclass for dataloaders
        parent_dict = deepcopy(self.dl.cfg_dict())
        if self._max_syscalls is not None:
            parent_dict["max_syscalls_training"] = self._max_syscalls_training
            parent_dict["max_syscalls_validation"] = self._max_syscalls_validation
        return parent_dict

    def metrics(self):
        parent_dict = deepcopy(self.dl.metrics())
        parent_dict["training_syscalls"] = self._get_syscall_counter(self._applied_training_filters)
        parent_dict["val_syscalls"] = self._get_syscall_counter(self._applied_val_filters)
        return parent_dict

    def artifact_dict(self):
        return self.dl.artifact_dict()

    def _get_syscall_counter(self, filters: List[MaxSyscallFilter]):
        syscalls = None
        for f in filters:
            if syscalls is None:
                syscalls = f._syscall_counter
                continue
            if syscalls != f._syscall_counter:
                raise RuntimeError("Expected the loaded syscalls to be equal for each filter! %s != %s" % (
                syscalls, f._syscall_counter))
        return syscalls

    def get_val_ratio(self):
        return self._validation_ratio