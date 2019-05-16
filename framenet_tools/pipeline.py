import logging
from typing import List

from framenet_tools.config import ConfigManager
from framenet_tools.data_handler.rawreader import RawReader
from framenet_tools.data_handler.semevalreader import SemevalReader
from framenet_tools.stages.feeID import FeeID
from framenet_tools.stages.frameID import FrameID
from framenet_tools.stages.spanID import SpanID

stage_names = ["feeID",
               "frameID",
               "spanID",
               "roleID"]


def get_stages(i: int, cM: ConfigManager):
    """
    Creates a list of stages up to the bound specified

    :param i: The upper bound of the pipeline stages
    :return: A list of stages
    """

    stages = [
        FeeID(cM),
        FrameID(cM),
        SpanID(cM),
        # RoleID(cM)
    ]

    return stages[:i]


class Pipeline(object):

    def __init__(self, cM: ConfigManager, level):
        self.cM = cM
        self.level = level

        self.stages = get_stages(self.level, cM)

    def train(self, data: List[str]):

        reader, reader_dev = self.load_dataset()

        for stage in self.stages:
            stage.train(reader, reader_dev)

    def predict(self, file: str, out_path: str):

        m_reader = RawReader(self.cM)

        m_reader.read_raw_text(file)

        for stage in self.stages:
            stage.predict(m_reader)

        m_reader.export_to_json(out_path)

    def load_dataset(self, files: List[str] = None):

        file = "data/experiments/xp_001/data/train.gold.xml"

        m_data_reader = SemevalReader(self.cM)
        m_data_reader.read_data(file)

        file = self.cM.semeval_files[0]
        m_data_reader_dev = SemevalReader(self.cM)
        m_data_reader_dev.read_data(file)

        return m_data_reader, m_data_reader_dev

    def evaluate(self):
        """

        :return:
        """

        # TODO


logging.basicConfig(
        format="%(asctime)s-%(levelname)s-%(message)s", level=logging.INFO
    )

#cM = ConfigManager()

#p = Pipeline(cM)
#p.train(cM.train_files)
#p.predict("example.txt", "test.json")

