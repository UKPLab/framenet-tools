from typing import List

from framenet_tools.config import ConfigManager
from framenet_tools.data_handler.reader import DataReader
from framenet_tools.frame_identification.frameidentifier import FrameIdentifier
from framenet_tools.pipelinestage import PipelineStage


class FrameID(PipelineStage):

    def __init__(self, cM: ConfigManager):
        super().__init__(cM)

        self.f_i = FrameIdentifier(cM)

    def train(self, data: List[str]):

        self.f_i.train(data)

        self.f_i.save_model(self.cM.saved_model)

    def predict(self, m_reader: DataReader):

        self.f_i.load_model(self.cM.saved_model)

        for annotations in m_reader.annotations:
            for annotation in annotations:
                frame = self.f_i.query(annotation)

                annotation.frame = frame