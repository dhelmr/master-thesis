import os
import tempfile

import mlflow
from pandas import DataFrame

from algorithms.building_block import BuildingBlock
from algorithms.features.impl.max_score_threshold import MaxScoreThreshold
from algorithms.ids import IDS
from dataloader.syscall import Syscall
from tsa.analysis.analyser import AnalyserBB
from tsa.building_block_builder import IDSPipelineBuilder
from tsa.experiment import Experiment

class DummyDecider(BuildingBlock):
    def __init__(self, inp: BuildingBlock):
        super().__init__()
        self._input = inp
        self._deps = [inp]

    def _calculate(self, syscall: Syscall):
        return 0

    def is_decider(self):
        return True
    def depends_on(self) -> list:
        return self._deps

class AnalysisExperiment(Experiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_test(self, dataloader, run_cfg):
        ids_cfg = self._get_param("ids", exp_type=list)
        builder = IDSPipelineBuilder()
        last_bb = builder.build_all(ids_cfg)

        decider = DummyDecider(last_bb)

        ids = IDS(data_loader=dataloader,
                  resulting_building_block=decider,
                  create_alarms=True,
                  plot_switch=False)

        for bb in unpack_dependencies(last_bb):
            if isinstance(bb, AnalyserBB):
                stats = bb.get_stats()
                if isinstance(stats, DataFrame):
                    self.log_pandas_df(stats, name=bb.name)
                elif stats is None:
                    print("Got no stats")
                else:
                    print("Unknown stats type!", stats.__class__)

        return {}, {}, ids

    def log_pandas_df(self, df: DataFrame, name: str):
        tmpdir = tempfile.mkdtemp()
        tmpfile = os.path.join(tmpdir, f"{name}.parquet.gz")
        df.to_parquet(tmpfile, compression="gzip")
        mlflow.log_artifact(tmpfile)

def unpack_dependencies(last_bb: BuildingBlock):
    inner_deps = []
    for dep in last_bb.depends_on():
        inner_deps += unpack_dependencies(dep)
    return inner_deps + [last_bb]
