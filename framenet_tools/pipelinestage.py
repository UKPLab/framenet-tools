from abc import ABC, abstractmethod

from framenet_tools.config import ConfigManager
from framenet_tools.data_handler.reader import DataReader


class PipelineStage(ABC):
    """
    Abstract stage of the pipeline
    """

    @abstractmethod
    def __init__(self, cM: ConfigManager):
        self.cM = cM
        self.loaded = False

    @abstractmethod
    def train(self, m_reader: DataReader, m_reader_dev: DataReader):
        """
        Train the stage on the given data

        :param m_reader: The DataReader object which contains the training data
        :param m_reader_dev: The DataReader object for evaluation and auto stopping
                            (NOTE: not necessarily given, as the focus might lie on maximizing the training data)
        :return:
        """

    @abstractmethod
    def predict(self, mReader: DataReader):
        """
        Predict the given data

        NOTE: Changes the object itself

        :param mReader: The DataReader object
        :return:
        """