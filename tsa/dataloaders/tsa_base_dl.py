from dataloader.base_data_loader import BaseDataLoader

class TsaBaseDataloader(BaseDataLoader):
    def cfg_dict(self):
        raise NotImplementedError()

    def metrics(self):
        raise NotImplementedError()

    def artifact_dict(self):
        raise NotImplementedError()
