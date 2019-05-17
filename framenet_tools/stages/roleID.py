from framenet_tools.config import ConfigManager
from framenet_tools.data_handler.reader import DataReader
from framenet_tools.pipelinestage import PipelineStage
from framenet_tools.role_identification.roleidentifier import RoleIdentifier


class RoleID(PipelineStage):
    """
    The Role Identification stage
    """

    def __init__(self, cM: ConfigManager):
        super().__init__(cM)

        self.r_i = RoleIdentifier(cM)

    def train(self, m_reader: DataReader, m_reader_dev: DataReader):
        """
        Trains the role identification stage

        :param m_reader: The DataReader object which contains the training data
        :param m_reader_dev: The DataReader object for evaluation and auto stopping
                            (NOTE: not necessarily given, as the focus might lie on maximizing the training data)
        :return:
        """

        self.r_i.train(m_reader, m_reader_dev)

    def predict(self, m_reader: DataReader):
        """

        :param m_reader:
        :return:
        """
