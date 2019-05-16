from framenet_tools.data_handler.reader import DataReader
from framenet_tools.utils.utils import get_sentences
from framenet_tools.config import ConfigManager


class RawReader(DataReader):
    def __init__(
        self, cM: ConfigManager, raw_path: str = None
    ):

        DataReader.__init__(self, cM)

        self.raw_path = raw_path

    def read_raw_text(self, raw_path: str = None):
        """
        Reads a raw text file and saves the content as a dataset

        NOTE: Applying this function removes the previous dataset content

        :param raw_path: The path of the file to read
        :return:
        """

        if raw_path is not None:
            self.raw_path = raw_path

        if self.raw_path is None:
            raise Exception("Found no file to read")

        file = open(raw_path, "r")
        raw = file.read()
        file.close()

        self.sentences += get_sentences(raw, self.cM.use_spacy)

        self.loaded(False)

