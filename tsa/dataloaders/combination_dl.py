from typing import List

from dataloader.base_data_loader import BaseDataLoader
from tsa.dataloaders.tsa_base_dl import TsaBaseDataloader


def yield_successively(lists, limit=None):
    cur_indexes = [0] * len(lists)
    yielded_one = True
    count = 0
    while yielded_one:
        yielded_one = False
        for i, l in enumerate(lists):
            if cur_indexes[i] >= len(l):
                continue
            yield l[cur_indexes[i]]
            cur_indexes[i] += 1
            yielded_one = True
            count += 1
            if limit is not None and count >= limit:
                return


class CombinationDL(TsaBaseDataloader):
    def __init__(self, dls: List[TsaBaseDataloader]):
        super().__init__("+".join([dl.scenario_path for dl in dls]))
        self._dls = dls

    def _combine(self, recordings: list) -> list:
        return list(yield_successively(recordings))

    def training_data(self) -> list:
        training_data = [dl.training_data() for dl in self._dls]
        print([dl._training_size for dl in self._dls])
        return self._combine(training_data)

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

    def artifact_dict(self):
        artifacts = {}
        for i, dl in enumerate(self._dls):
            dl_artifacts = dl.artifact_dict()
            for key, val in dl_artifacts.items():
                artifacts["%s.%s" % (i, key)] = val
        return artifacts

    def metrics(self):
        metrics = {}
        for i, dl in enumerate(self._dls):
            dl_metrics = dl.metrics()
            for key, val in dl_metrics.items():
                metrics["%s.%s" % (i, key)] = val
        return metrics