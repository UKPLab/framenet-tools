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

    :param i:
    :return:
    """

    stages = [
        FeeID(cM),
        FrameID(cM),
        SpanID(cM),
        # RoleID(cM)
    ]

    return stages


class Pipeline(object):

    def __init__(self, cM: ConfigManager):
        self.cM = cM
        self.level = cM.level

        self.stages = get_stages(self.level, cM)

    def train(self, data: List[str]):

        reader, reader_dev = self.load_dataset()

        for stage in self.stages:
            stage.train(reader, reader_dev)

    def predict(self, file: List[str]):

        m_reader = RawReader(self.cM)

        m_reader.read_raw_text(file[0])

        for stage in self.stages:
            stage.predict(m_reader)

        m_reader.export_to_json("test.json")

    def load_dataset(self, files: List[str] = None):

        file = "data/experiments/xp_001/data/train.gold.xml"

        m_data_reader = SemevalReader(cM)
        m_data_reader.read_data(file)

        file = cM.semeval_files[0]
        m_data_reader_dev = SemevalReader(cM)
        m_data_reader_dev.read_data(file)

        return m_data_reader, m_data_reader_dev


logging.basicConfig(
        format="%(asctime)s-%(levelname)s-%(message)s", level=logging.INFO
    )

cM = ConfigManager()

p = Pipeline(cM)
#p.train(cM.train_files)
p.predict(["example.txt"])

