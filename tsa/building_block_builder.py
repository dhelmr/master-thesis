from typing import List

from algorithms.building_block import BuildingBlock
from algorithms.decision_engines.ae import AE
from algorithms.decision_engines.scg import SystemCallGraph
from algorithms.decision_engines.som import Som
from algorithms.decision_engines.stide import Stide
from algorithms.features.impl.int_embedding import IntEmbedding
from algorithms.features.impl.max_score_threshold import MaxScoreThreshold
from algorithms.features.impl.ngram import Ngram as _Ngram
from algorithms.features.impl.one_hot_encoding import OneHotEncoding
from algorithms.features.impl.stream_sum import StreamSum
from algorithms.features.impl.syscall_name import SyscallName
from algorithms.features.impl.w2v_embedding import W2VEmbedding
from tsa.FrequencySTIDE import FrequencySTIDE
from tsa.analysis.analyser import TrainingSetAnalyser
from tsa.analysis.ngram_analyser import NgramAnalyser
from tsa.analysis.continuous_anaylser import ContinuousTrainingSetAnalyser
from tsa.analysis.visualization_analyser import Visualize
from tsa.dataloaders.training_set_filter import TrainingSetFilter
from tsa.frequency_encoding import FrequencyEncoding
from tsa.ngram_frequency_append import NgramFrequencyAppender
from tsa.unsupervised.pca import PCA_BB
from tsa.unsupervised.scikit import LOF, EllipticEnvelopeOD, IsolationForestOD
from tsa.unsupervised.frequency_od import FrequencyOD
from tsa.unsupervised.mixed_model import MixedModelOutlierDetector
from tsa.unsupervised.w2v_concat import W2VConcat, TupleBB
from tsa.utils import access_cfg, exists_key


def Ngram(building_block, *args, **kwargs):
    return _Ngram([building_block], *args, **kwargs)

def Ngram(building_block, *args, **kwargs):
    return _Ngram([building_block], *args, **kwargs)

BUILDING_BLOCKS = {cls.__name__: cls for cls in
                   [AE, Stide, Som, SystemCallGraph, IntEmbedding, W2VEmbedding, OneHotEncoding, Ngram, LOF,
                    MixedModelOutlierDetector, MaxScoreThreshold, StreamSum, FrequencyOD,
                    EllipticEnvelopeOD, IsolationForestOD,
                    TrainingSetAnalyser, ContinuousTrainingSetAnalyser, Visualize, NgramAnalyser,
                    PCA_BB, W2VConcat, TupleBB, TrainingSetFilter, FrequencyEncoding, FrequencySTIDE,
                    NgramFrequencyAppender]}
BuildingBlockCfg = dict


class IDSPipelineBuilder:
    def __init__(self):
        self.analysers = []

    def _build_block(self, cfg: BuildingBlockCfg, last_block: BuildingBlock) -> BuildingBlock:
        print(cfg)
        name = access_cfg(cfg, "name")
        if name not in BUILDING_BLOCKS:
            raise ValueError("%s is not a valid BuildingBlock name" % name)
        bb_class = BUILDING_BLOCKS[name]
        bb_args = access_cfg(cfg, "args", default={})
        if exists_key(cfg, "dependencies"):
            arg_name = access_cfg(cfg, "dependencies", "name")
            cfg_list = access_cfg(cfg, "dependencies", "blocks", required=True)
            dependency_bb = self.build_all(cfg_list)
            bb_args[arg_name] = dependency_bb
        try:
            bb = bb_class(last_block, **bb_args)
        except Exception as e:
            raise RuntimeError("Error building block %s with args %s.\nNested Error is: %s" % (name, cfg, e)) from e
        return bb

    def build_all(self, configs: List[BuildingBlockCfg]):
        last_block = SyscallName()
        for i, cfg in enumerate(configs):
            last_block = self._build_block(cfg, last_block)
        return last_block
