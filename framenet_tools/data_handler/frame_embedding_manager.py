import logging


class FrameEmbeddingManager(object):
    def __init__(
        self, path: str = "data/frame_embeddings/dict_frame_to_emb_100dim_wsb_list.txt"
    ):

        self.path = path
        self.frames = dict()

    def read_frame_embeddings(self):
        """
        Loads the previously specified frame embedding file into a dictionary
        """

        logging.info("Loading frame embeddings")

        file = open(self.path, "r")
        raw = file.read()
        file.close()

        data = raw.rsplit("\n")

        for line in data:
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
