from framenet_tools.config import ConfigManager
from framenet_tools.data_handler.reader import DataReader
from framenet_tools.frame_identification.frameidentifier import FrameIdentifier
from framenet_tools.pipelinestage import PipelineStage


class FrameID(PipelineStage):
    """
    The Frame Identification stage
    """

    def __init__(self, cM: ConfigManager):
        super().__init__(cM)

        self.f_i = FrameIdentifier(cM)

    def train(self, m_reader: DataReader, m_reader_dev: DataReader):
        """
        Train the frame identification stage on the given data

        NOTE: May overwrite a previously saved model!

        :param m_reader: The DataReader object which contains the training data
        :param m_reader_dev: The DataReader object for evaluation and auto stopping
                            (NOTE: not necessarily given, as the focus might lie on maximizing the training data)
        :return:
        """

        self.f_i.train(m_reader, m_reader_dev)

        self.f_i.save_model(self.cM.saved_model)

    def predict(self, m_reader: DataReader):

        self.f_i.load_model(self.cM.saved_model)

        for annotations in m_reader.annotations:
            for annotation in annotations:
                frame = self.f_i.query(annotation)
                frames = self.f_i.query_confidence(annotation)

                annotation.frame = frame
                annotation.frame_confidence = frames
