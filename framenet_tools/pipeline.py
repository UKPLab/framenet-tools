import logging
from copy import deepcopy
from typing import List

from framenet_tools.config import ConfigManager
from framenet_tools.data_handler.rawreader import RawReader
from framenet_tools.data_handler.semevalreader import SemevalReader
from framenet_tools.evaluator import evaluate_stages
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
        SpanID(cM),
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

        if not levels:
            self.levels = [0, 1, 2]

        self.stages = [get_stages(level, cM) for level in self.levels]

    def train(self, data: List[str], dev_data: List[str] = None):
        """
        Trains all stages up to the specified level

        :param data: The data to train on
        :param dev_data: The data to check evaluation on
        :return:
        """

        if not self.levels:
            logging.info(f"NOTE: Training an empty pipeline!")

        reader = self.load_dataset(data)
        reader_dev = None

        if dev_data is not None:
            reader_dev = self.load_dataset(dev_data)

        for stage in self.stages:
            stage.train(reader, reader_dev)

    def predict(self, file: str, out_path: str):
        """
        Predicts a raw file and exports the predictions to the given file.
        Also only predicts up to the specified level.

        NOTE: Prediction is only possible up to the level on which the pipeline was trained!

        :param file: The raw input text file
        :param out_path: The path to save the outputs to (can be None)
        :return:
        """

        m_reader = RawReader(self.cM)

        m_reader.read_raw_text(file)

        for stage in self.stages:
            stage.predict(m_reader)

        m_reader.export_to_json(out_path)

        logging.info(f"Prediction successful!")

    def load_dataset(self, files: List[str]):
        """
        Helper function for loading datasets.

        :param files: A List of files to load the datasets from.
        :return: A reader object containing the loaded data.
        """

        m_data_reader = SemevalReader(self.cM)
        for file in files:
            m_data_reader.read_data(file)

        return m_data_reader

    def evaluate(self):
        """
        Evaluates all the specified stages of the pipeline.

        NOTE: Depending on the certain levels of the pipeline, the propagated error can be large!

        :return:
        """

        for file in self.cM.semeval_dev + self.cM.semeval_test:

            logging.info(f"Evaluation on {file}:")

            m_reader = self.load_dataset([file])
            original_reader = deepcopy(m_reader)

            for stage in self.stages:
                stage.predict(m_reader)

            evaluate_stages(m_reader, original_reader, self.levels)
