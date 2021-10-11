class BaseStreamFeatureExtractor:
    """

    base class for feature extraction from a stream of system call features

    """

    def __init__(self):
        pass

    def train_on(self, syscall_feature: dict):
        """

        takes features of one system call to train this extraction

        """
        pass

    def fit(self):
        """

        finalizes training

        """
        pass

    def extract(self, syscall_features: dict) -> dict:
        """

        extracts a feature from a stream of syscall features

        Returns:
        dict: key: name of feature and
              value: value of feature

        """
        pass
