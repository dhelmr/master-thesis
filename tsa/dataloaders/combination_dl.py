from typing import List

from dataloader.base_data_loader import BaseDataLoader
from tsa.dataloaders.tsa_base_dl import TsaBaseDataloader


def yield_successively(lists):
    cur_indexes = [0] * len(lists)
    yielded_one = True
    while yielded_one:
        yielded_one = False
        for i, l in enumerate(lists):
            if cur_indexes[i] >= len(l):
                continue
            yield l[cur_indexes[i]]
            cur_indexes[i] += 1
            yielded_one = True


class CombinationDL(TsaBaseDataloader):
    def __init__(self, dls: List[TsaBaseDataloader]):
        super().__init__("+".join([dl.scenario_path for dl in dls]))
        self._dls = dls

    def _combine(self, recordings: list) -> list:
        return list(yield_successively(recordings))

    def training_data(self) -> list:
        return self._combine([dl.training_data() for dl in self._dls])

    def validation_data(self) -> list:
        return self._combine([dl.validation_data() for dl in self._dls])

    def test_data(self) -> list:
        return self._combine([dl.test_data() for dl in self._dls])

    def extract_recordings(self, category: str) -> list:
        raise NotImplementedError("not implemented")

    def collect_metadata(self) -> dict:
        aggregated_metadata = {}
        for dl in self._dls:
            aggregated_metadata.update(dl.collect_metadata())
        return aggregated_metadata

    def get_direction_string(self):
        direction_str = None
        for dl in self._dls:
            cur_direction_str = dl.get_direction_string()
            if direction_str is None:
                direction_str = cur_direction_str
            elif direction_str != cur_direction_str:
                raise RuntimeError("Direction strings of the dataloaders do not match! %s != %s" % (cur_direction_str, direction_str))
        return direction_str

    def cfg_dict(self):
        # TODO implement own superclass for dataloaders
        #parent_dict = deepcopy(self.dl.cfg_dict())
        #parent_dict["max_syscalls"] = self._max_syscalls
        aggregated_cfg = {}
        for dl in self._dls:
            aggregated_cfg.update(dl.cfg_dict())
        return aggregated_cfg

    def get_val_ratio(self):
        val_ratio = None
        for dl in self._dls:
            if val_ratio is None:
                val_ratio = dl.get_val_ratio()
            elif val_ratio != dl.get_val_ratio():
                raise RuntimeError("Validation error must be equal for all dataloaders")
        if val_ratio is None:
            raise RuntimeError("No dataloaders found")
        return val_ratio