from typing import List

from framenet_tools.config import ConfigManager
from framenet_tools.data_handler.reader import DataReader
from framenet_tools.pipelinestage import PipelineStage
from framenet_tools.span_identification.spanidentifier import SpanIdentifier


class SpanID(PipelineStage):

    def __init__(self, cM: ConfigManager):
        super().__init__(cM)

        self.s_i = SpanIdentifier(cM)

    def train(self, data: List[str]):


        self.s_i.train(mReader, mReaderDev)

    def predict(self, m_reader: DataReader):
        """

        :param m_reader:
        :return:
        """
