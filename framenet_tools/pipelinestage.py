from abc import ABC, abstractmethod
from typing import List

from framenet_tools.config import ConfigManager
from framenet_tools.data_handler.annotation import Annotation
from framenet_tools.data_handler.reader import DataReader


class PipelineStage(ABC):

    @abstractmethod
    def __init__(self, cM: ConfigManager):
        self.cM = cM

    @abstractmethod
    def train(self, data: List[str]):
        """
        Train the stage on the given data

        :param data:
        :return:
        """

    @abstractmethod
    def predict(self, mReader: DataReader):
        """
        Predict the given data

        NOTE: Changes the object itself

        :param data:
        :return:
        """