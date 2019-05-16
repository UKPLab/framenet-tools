import logging

from tqdm import tqdm


class FrameEmbeddingManager(object):
    """
    Loads and provides the specified frame-embeddings
    """

    def __init__(
        self, path: str = "data/frame_embeddings/dict_frame_to_emb_100dim_wsb_list.txt"
    ):

        self.path = path
        self.frames = None

    def string_to_array(self, string: str):
        """
        Helper function
        Converts a string of an array back into an array

        NOTE: specified for float arrays !!!

        :param string: The string of an array
        :return: The array
        """

        array = []

        string = string.replace('[', '')
        string = string.replace(']', '')
        string = string.rsplit(',')

        for element in string:
            array.append(float(element))

        return array

    def read_frame_embeddings(self):
        """
        Loads the previously specified frame embedding file into a dictionary
        """

        if self.frames is not None:
            return

        self.frames = dict()

        logging.info("Loading frame embeddings")

        with open(self.path, "r") as file:
            raw = file.read()

        data = raw.rsplit("\n")

        for line in tqdm(data):
            line = line.rsplit("\t")

            if len(line) > 1:
                self.frames[line[0]] = self.string_to_array(line[1])

        logging.info("[Done] loading frame embeddings")

    def embed(self, frame: str):
        """
        Converts a given frame to its embedding

        :param frame: The frame to embed
        :return: The embedding (n-dimensional vector)
        """

        if frame in self.frames:
            return self.frames[frame]

        return None
