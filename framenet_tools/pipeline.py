import logging
from typing import List

from framenet_tools.config import ConfigManager
from framenet_tools.data_handler.rawreader import RawReader
from framenet_tools.data_handler.semevalreader import SemevalReader
from framenet_tools.stages.feeID import FeeID
from framenet_tools.stages.frameID import FrameID
from framenet_tools.stages.spanID import SpanID

stage_names = ["feeID", "frameID", "spanID", "roleID"]


def get_stages(i: int, cM: ConfigManager):
    """
    Creates a list of stages up to the bound specified

    :param i: The upper bound of the pipeline stages
    :return: A list of stages
    """

    stages = [
        FeeID(cM),
        FrameID(cM),
        # SpanID(cM),
        # RoleID(cM)
    ]

    return stages[i]


class Pipeline(object):
    """
    The SRL pipeline

    Contains the stages of Frame evoking element identification, Frame identification,
    Span identification and Role identification.
    """

    def __init__(self, cM: ConfigManager, levels: List[int]):
        self.cM = cM
        self.levels = levels

        self.stages = [get_stages(level, cM) for level in levels]

    def train(self, data: List[str]):
        """
        Trains all stages up to the specified level

        :param data: The data to train on TODO
        :return:
        """

        if self.levels == []:
            logging.info(f"NOTE: Training an empty pipeline!")

        reader, reader_dev = self.load_dataset()

        for stage in self.stages:
            stage.train(reader, reader_dev)

    def predict(self, file: str, out_path: str):
        """
        Predicts a raw file and exports the predictions to the given file.
        Also only predicts up to the specified level.

        NOTE: Prediction is only possible up to the level on which the pipeline was trained!

        :param file: The raw input text file
        :param out_path: The path to save the outputs to
        :return:
        """

        m_reader = RawReader(self.cM)

        m_reader.read_raw_text(file)

        for stage in self.stages:
            stage.predict(m_reader)

        m_reader.export_to_json(out_path)

    def load_dataset(self, files: List[str] = None):
        """
        Helper function for loading datasets

        :param files:
        :return:
        """

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
