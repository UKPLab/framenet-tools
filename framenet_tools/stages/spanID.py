from framenet_tools.config import ConfigManager
from framenet_tools.data_handler.reader import DataReader
from framenet_tools.pipelinestage import PipelineStage
from framenet_tools.span_identification.spanidentifier import SpanIdentifier


class SpanID(PipelineStage):
    """
    The Span Identification stage
    """

    def __init__(self, cM: ConfigManager):
        super().__init__(cM)

        self.s_i = SpanIdentifier(cM)

    def train(self, m_reader: DataReader, m_reader_dev: DataReader):
        """
        Train the stage on the given data

        :param m_reader: The DataReader object which contains the training data
        :param m_reader_dev: The DataReader object for evaluation and auto stopping
                            (NOTE: not necessarily given, as the focus might lie on maximizing the training data)
        :return:
        """

        m_reader.generate_pos_tags()

        m_reader.embed_words()
        m_reader.embed_frames()

        m_reader_dev.generate_pos_tags()

        m_reader_dev.embed_words()
        m_reader_dev.embed_frames()

        self.s_i.train(m_reader, m_reader_dev)

    def predict(self, m_reader: DataReader):
        """

        :param m_reader:
        :return:
        """

        self.s_i.load()

        m_reader.generate_pos_tags()

        m_reader.embed_words()
        m_reader.embed_frames()

        self.s_i.predict_spans(m_reader)
