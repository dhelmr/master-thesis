from algorithms.features.impl.int_embedding import IntEmbedding
from algorithms.features.impl.threadID import ThreadID
from algorithms.features.impl.ngram import Ngram
# from algorithms.features.time_delta_syscalls import TimeDeltaSyscalls
# from algorithms.features.thread_change_flag import ThreadChangeFlag

from algorithms.decision_engines.stide import Stide
from algorithms.ids import IDS

from dataloader.dataloader_real_world import DataLoaderRealWorld
from dataloader.direction import Direction

from pprint import pprint

if __name__ == '__main__':
    """
    this is an example script to show the usage uf our classes
    """
    ngram_length = 3
    embedding_size = 4
    thread_aware = True

    path='../../WHK/Data/real_world/'
    # data loader for scenario
    dataloader = DataLoaderRealWorld(path, direction=Direction.CLOSE)

    # embedding
    int_embedding = IntEmbedding()

    # extra parameter

    # ngrams
    ngram = Ngram(feature_list=[int_embedding],
                  thread_aware=True,
                  ngram_length=ngram_length + 1
                 )

    # decision engine (DE)
    de = Stide(ngram)

    # define the used features
    ids = IDS(data_loader=dataloader,
              resulting_building_block=de,
              plot_switch=False)

    # threshold
    ids.determine_threshold()
    # detection
    ids.do_detection()
    # print(results)
    pprint(ids.performance.get_performance())
