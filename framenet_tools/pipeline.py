import logging
from typing import List

from framenet_tools.config import ConfigManager
from framenet_tools.data_handler.reader import DataReader
from framenet_tools.stages.feeID import FeeID
from framenet_tools.stages.frameID import FrameID

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
        # spanID(cM),
        # roleID(cM)
    ]

    return stages




class Pipeline(object):

    def __init__(self, cM: ConfigManager):
        self.cM = cM
        self.level = cM.level

        self.stages = get_stages(self.level, cM)

    def train(self, data: List[str]):

        for stage in self.stages:
            stage.train(data)

    def predict(self, file: List[str]):

        m_reader = DataReader(self.cM)
        #if len(file) == 2:
        #    reader.read_data(file[0], file[1])
        #else:
        m_reader.read_raw_text(file[0])

        for stage in self.stages:
            stage.predict(m_reader)

        m_reader.predict_spans()
        m_reader.export_to_json("test.json")


logging.basicConfig(
        format="%(asctime)s-%(levelname)s-%(message)s", level=logging.INFO
    )

cM = ConfigManager()

p = Pipeline(cM)
# p.train(cM.train_files)
p.predict(["example.txt"])

