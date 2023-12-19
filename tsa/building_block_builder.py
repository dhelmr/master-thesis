import copy
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
from tsa.CacheableW2v import CacheableW2V
from tsa.accommodation.FrequencySTIDE import FrequencySTIDE
from tsa.accommodation.MicroSTIDEs import MicroSTIDEs
from tsa.accommodation.NgramThreadDistance import NgramThreadDistance
from tsa.accommodation.NgramThreadEntropy import NgramThreadEntropy
from tsa.analysis.analyser import TrainingSetAnalyser
from tsa.analysis.cluster_visualization_analyser import ClusterVisualize
from tsa.analysis.data_drift_analyser import DataDriftAnalyser
from tsa.analysis.frequency_distribution import FrequencyDistribution
from tsa.analysis.ngram_analyser import NgramAnalyser
from tsa.analysis.continuous_anaylser import ContinuousTrainingSetAnalyser
from tsa.analysis.ngram_thread_analyser import NgramThreadAnalyser
from tsa.analysis.visualization_analyser import Visualize
from tsa.dataloaders.training_set_filter import TrainingSetFilter
from tsa.accommodation.frequency_encoding import FrequencyEncoding
from tsa.accommodation.ngram_frequency_append import NgramFrequencyAppender
from tsa.accommodation.percentile_threshold import PercentileThreshold
from tsa.accommodation.score_mult import ScoreMultiplication
from tsa.accommodation.ngram_thread_pca import NgramThreadEmbeddingBB
from tsa.accommodation.tfidf_stide import TfidfSTIDE
from tsa.diagnosis.pca import PCA_BB
from tsa.diagnosis.scikit import LOF, EllipticEnvelopeOD, IsolationForestOD
from tsa.diagnosis.frequency_od import FrequencyOD
from tsa.diagnosis.mixed_model import MixedModelOutlierDetector
from tsa.diagnosis.thread_clustering import ThreadClusteringOD
from tsa.diagnosis.w2v_concat import W2VConcat, TupleBB
from tsa.utils import access_cfg, exists_key


def Ngram(building_block, *args, **kwargs):
    return _Ngram([building_block], *args, **kwargs)

def Ngram(building_block, *args, **kwargs):
    return _Ngram([building_block], *args, **kwargs)

BUILDING_BLOCKS = {cls.__name__: cls for cls in
                   [AE, Stide, Som, SystemCallGraph, IntEmbedding, W2VEmbedding, CacheableW2V, OneHotEncoding, Ngram, LOF,
                    MixedModelOutlierDetector, MaxScoreThreshold, StreamSum, FrequencyOD,
                    EllipticEnvelopeOD, IsolationForestOD,
                    TrainingSetAnalyser, ContinuousTrainingSetAnalyser, Visualize, NgramAnalyser,
                    PCA_BB, W2VConcat, TupleBB, TrainingSetFilter, FrequencyEncoding, FrequencySTIDE,
                    NgramFrequencyAppender, ThreadClusteringOD, NgramThreadEntropy, ScoreMultiplication, NgramThreadEmbeddingBB,
                    NgramThreadAnalyser, NgramThreadDistance, ClusterVisualize, TfidfSTIDE, FrequencyDistribution, MicroSTIDEs,
                    PercentileThreshold, DataDriftAnalyser]}
BuildingBlockCfg = dict


class IDSPipelineBuilder:
    def __init__(self, cache_context):
        self.analysers = []
        self._cache_context = cache_context

    def _build_block(self, cfg: BuildingBlockCfg, last_block: BuildingBlock, cache_key) -> BuildingBlock:
        cfg = copy.deepcopy(cfg)
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
        if access_cfg(cfg, "cache_key", default=False):
            bb_args["cache_key"] = cache_key
        try:
            bb = bb_class(last_block, **bb_args)
        except Exception as e:
            raise RuntimeError("Error building block %s with args %s.\nNested Error is: %s" % (name, cfg, e)) from e
        return bb

    def _build_split_block(self, cfg: BuildingBlockCfg, last_block: BuildingBlock, cache_key):
        parallel_blocks = []
        for _, config_list in cfg["split"].items():
            bb = self._build_bb_pipeline(config_list, cache_key, last_block)
            parallel_blocks.append(bb)
        return parallel_blocks

    def build_all(self, configs: List[BuildingBlockCfg]):
        first_block = SyscallName()
        last_block = self._build_bb_pipeline(configs,
                                             cache_key=self._cache_context,
                                             from_block=first_block)
        return last_block


    def _build_bb_pipeline(self, configs: List[BuildingBlockCfg], cache_key: str, from_block: BuildingBlock):
        last_block = from_block
        for i, cfg in enumerate(configs):
            cache_key += "||" + str(cfg)
            print(cfg)
            if exists_key(cfg, "split"):
                last_block = self._build_split_block(cfg, last_block, cache_key)
            else:
                last_block = self._build_block(cfg, last_block, cache_key)
        return last_block
