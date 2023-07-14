import hashlib
import os
import pickle

from gensim.models import Word2Vec

from algorithms.building_block import BuildingBlock
from dataloader.syscall import Syscall

from algorithms.features.impl.w2v_embedding import W2VEmbedding
from tsa.frequency_encoding import FrequencyAnomalyFunction
from tsa.unsupervised.mixed_model import Histogram


class CacheableW2V(W2VEmbedding):

    def __init__(self, input: BuildingBlock, cache_key: str = None, *args, **kwargs):
        super().__init__(input, *args, **kwargs)
        self._cache_key = cache_key
        if self._cache_key is not None:
            self.w2vmodel = self._load_model_from_cache(cache_key)

    def fit(self):
        super().fit()
        if self._cache_key is not None:
            self.write_model_to_cache(self._cache_key)

    def _load_model_from_cache(self, key):
        model_path = self._cache_key_to_path(key)
        if not os.path.exists(model_path):
            return None
        print("Load w2v model from %s" % model_path)
        return Word2Vec.load(model_path)

    def _cache_key_to_path(self, cache_key: str):
        if "W2V_CACHE_PATH" not in os.environ:
            raise KeyError("$W2V_CACHE_PATH must be set")

        md5_hash = hashlib.md5(cache_key.encode()).hexdigest()
        model_path = os.path.join(os.environ["W2V_CACHE_PATH"], "%s.w2v.pickle" % md5_hash)
        return model_path

    def write_model_to_cache(self, key):
        model_path = self._cache_key_to_path(key)
        if os.path.exists(model_path):
            print("w2v model is already serialized at %s, skip" % model_path)
            return
        print("Write w2v model to %s" % model_path)
        with open(model_path, "wb") as f:
            self.w2vmodel.save(f)
