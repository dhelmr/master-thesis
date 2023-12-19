from pandas import DataFrame

from algorithms.building_block import BuildingBlock, IDSPhase
from algorithms.ids import IDS
from dataloader.syscall import Syscall
from tsa.analysis.analyser import AnalyserBB
from tsa.experiment import Experiment
from tsa.utils import log_pandas_df


class DummyDecider(BuildingBlock):
    def __init__(self, inp: BuildingBlock):
        super().__init__()
        self._input = inp
        if not isinstance(inp, AnalyserBB):
            raise ValueError("Last Building Block must be Analyser.")
        self._activate_test_phase = inp.test_phase
        self._deps = [inp]

    def _calculate(self, syscall: Syscall):
        if self._activate_test_phase and self._ids_phase == IDSPhase.TEST:
            return self._input._calculate(syscall)
        else:
            return 0

    def is_decider(self):
        return True

    def depends_on(self) -> list:
        return self._deps


class AnalysisExperiment(Experiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_test(self, dataloader, run_cfg, builder):
        ids_cfg = self._get_param("ids", exp_type=list)
        last_bb = builder.build_all(ids_cfg)

        decider = DummyDecider(last_bb)

        ids = IDS(
            data_loader=dataloader,
            resulting_building_block=decider,
            create_alarms=True,
            plot_switch=False,
        )
        if decider._activate_test_phase:
            ids.detect()

        for bb in unpack_dependencies(last_bb):
            if isinstance(bb, AnalyserBB):
                stats = bb.get_stats()
                print(stats)
                if isinstance(stats, DataFrame):
                    log_pandas_df(stats, name=bb.name)
                elif stats is None:
                    print("Got no stats")
                else:
                    print("Unknown stats type!", stats.__class__)

        return {}, {}, ids


def unpack_dependencies(last_bb: BuildingBlock):
    inner_deps = []
    for dep in last_bb.depends_on():
        inner_deps += unpack_dependencies(dep)
    return inner_deps + [last_bb]
