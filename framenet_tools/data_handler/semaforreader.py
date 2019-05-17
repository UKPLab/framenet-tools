from framenet_tools.data_handler.reader import DataReader
from framenet_tools.config import ConfigManager
from framenet_tools.data_handler.annotation import Annotation


class SemaforReader(DataReader):
    """
    A reader for the Semafor ConLL format

    Inherits from DataReader
    """

    def __init__(
        self, cM: ConfigManager, path_sent: str = None, path_elements: str = None
    ):

        DataReader.__init__(self, cM)

        # Provides the ability to set the path at object creation (can also be done on load)
        self.path_sent = path_sent
        self.path_elements = path_elements

    def digest_raw_data(self, elements: list, sentences: list):
        """
        Converts the raw elements and sentences into a nicely structured dataset

        NOTE: This representation is meant to match the one in the "frames-files"

        :param elements: the annotation data of the given sentences
        :param sentences: the sentences to digest
        :return:
        """

        # Append sentences
        for sentence in sentences:
            words = sentence.split(" ")
            while "" in words:
                words.remove("")
            self.sentences.append(words)

        for element in elements:
            # Element data
            element_data = element.split("\t")

            frame = element_data[3]  # Frame
            fee = element_data[4]  # Frame evoking element
            position = element_data[5].rsplit("_")  # Position of word in sentence
            position = (int(position[0]), int(position[-1]))
            fee_raw = element_data[6].rsplit(" ")[
                0
            ]  # Frame evoking element as it appeared

            sent_num = int(element_data[7])  # Sentence number

            if sent_num >= len(self.annotations):
                self.annotations.append([])

            roles, role_positions = self.digest_role_data(element)

            self.annotations[sent_num].append(
                Annotation(
                    frame,
                    fee,
                    position,
                    fee_raw,
                    self.sentences[sent_num],
                    roles,
                    role_positions,
                )
            )

    def digest_role_data(self, element: str):
        """
        Parses a string of role information into the desired format

        :param element: The string containing the role data
        :return: A pair of two concurrent lists containing the roles and their spans
        """

        roles = []
        role_positions = []

        element_data = element.split("\t")
        c = 8

        while len(element_data) > c:
            role = element_data[c]
            role_position = element_data[c + 1]
            if ":" in role_position:
                role_position = role_position.rsplit(":")
                role_position = (role_position[0], role_position[1])
            else:
                role_position = (role_position, role_position)
            role_position = (int(role_position[0]), int(role_position[1]))

            role_positions.append(role_position)
            roles.append(role)

            c += 2

        return roles, role_positions

    def read_data(self, path_sent: str = None, path_elements: str = None):
        """
        Reads a the sentence and elements file and saves the content as a dataset

        NOTE: Applying this function removes the previous dataset content

        :param path_sent: The path to the sentence file
        :param path_elements: The path to the elements
        :return:
        """

        if path_sent is not None:
            self.path_sent = path_sent

        if path_elements is not None:
            self.path_elements = path_elements

        if self.path_sent is None:
            raise Exception("Found no sentences-file to read!")

        if self.path_elements is None:
            raise Exception("Found no elements-file to read!")

        with open(self.path_sent, "r") as file:
            sentences = file.read()

        with open(self.path_elements, "r") as file:
            elements = file.read()

        sentences = sentences.split("\n")
        elements = elements.split("\n")

        # Remove empty line at the end
        if elements[len(elements) - 1] == "":
            elements = elements[: len(elements) - 1]

        if sentences[len(sentences) - 1] == "":
            sentences = sentences[: len(sentences) - 1]

        self.digest_raw_data(elements, sentences)

        self.loaded(True)
