import logging

from tqdm import tqdm
from typing import List


class WordEmbeddingManager(object):
    def __init__(
        self, path: str = "data/word_embeddings/levy_deps_300.w2vt"
    ):

        self.path = path
        self.words = None

    def string_to_array(self, strings: List[str]):
        """
        Helper function
        Converts a string of an array back into an array

        NOTE: specified for float arrays !!!

        :param string: The string of an array
        :return: The array
        """

        array = []

        for element in strings:
            array.append(float(element))

        # print(array)
        return array

    def read_word_embeddings(self):
        """
        Loads the previously specified frame embedding file into a dictionary
        """

        if self.words is not None:
            return

        self.words = dict()

        logging.info("Loading word embeddings")

        with open(self.path, "r") as file:
            raw = file.read()

        data = raw.rsplit("\n")[1:]

        for line in tqdm(data):
            line = line.rsplit(" ")

            self.words[line[0]] = self.string_to_array(line[1:])

        logging.info("[Done] loading word embeddings")

    def embed(self, word: str):
        """
        Converts a given word to its embedding

        :param word: The word to embed
        :return: The embedding (n-dimensional vector)
        """

        if word in self.words:
            return self.words[word]

        return None
