import logging

from tqdm import tqdm
from typing import List

from framenet_tools.frame_identification.feeidentifier import FeeIdentifier
from framenet_tools.frame_identification.utils import download_resources, get_sentences
from framenet_tools.config import ConfigManager
from framenet_tools.role_identification.spanidentifier import SpanIdentifier
from framenet_tools.data_handler.annotation import Annotation


class DataReader(object):
    def __init__(
        self, cM: ConfigManager, path_sent: str = None, path_elements: str = None, raw_path: str = None
    ):

        self.cM = cM

        # Provides the ability to set the path at object creation (can also be done on load)
        self.path_sent = path_sent
        self.path_elements = path_elements
        self.raw_path = raw_path

        self.sentences = []
        self.annotations = []

        # Flags
        self.is_annotated = None
        self.is_loaded = False

        self.dataset = []

        download_resources()

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
            if "" in words:
                words.remove("")
            self.sentences.append(words)

        for element in elements:
            # Element data
            element_data = element.split("\t")

            frame = element_data[3]  # Frame
            fee = element_data[4]  # Frame evoking element
            position = element_data[5]  # Position of word in sentence
            fee_raw = element_data[6]  # Frame evoking element as it appeared

            sent_num = int(element_data[7])  # Sentence number

            if sent_num >= len(self.annotations):
                self.annotations.append([])

            roles, role_positions = self.digest_role_data(element)

            self.annotations[sent_num].append(
                Annotation(frame, fee, position, fee_raw, self.sentences[sent_num], roles, role_positions)
            )

    def digest_role_data(self, element: str):
        """

        :param element:
        :return:
        """

        roles = []
        role_positions = []

        element_data = element.split("\t")
        c = 8

        while len(element_data) > c:
            role = element_data[c]
            role_position = element_data[c+1]
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

    def loaded(self, is_annotated: bool):
        """
        Helper for setting flags

        :param is_annotated: flag if loaded data was annotated
        :return:
        """

        self.is_loaded = True
        self.is_annotated = is_annotated

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
            raise Exception("Found no sentences-file to read")

        if self.path_elements is None:
            raise Exception("Found no elements-file to read")

        file = open(self.path_sent, "r")
        sentences = file.read()
        file.close()

        file = open(self.path_elements, "r")
        elements = file.read()
        file.close()

        sentences = sentences.split("\n")
        elements = elements.split("\n")

        # Remove empty line at the end
        if elements[len(elements) - 1] == "":
            # print("Removed empty line at eof")
            elements = elements[: len(elements) - 1]

        if sentences[len(sentences) - 1] == "":
            # print("Removed empty line at eof")
            sentences = sentences[: len(sentences) - 1]

        # print(sentences)

        self.digest_raw_data(elements, sentences)

        self.loaded(True)

    def predict_fees(self):
        """
        Predicts the Frame Evoking Elements
        NOTE: This drops current annotation data

        :return:
        """

        self.annotations = []
        fee_finder = FeeIdentifier(self.cM)

        for sentence in self.sentences:
            possible_fees = fee_finder.query([sentence])
            predicted_annotations = []

            # Create new Annotation for each possible frame evoking element
            for possible_fee in possible_fees:
                predicted_annotations.append(
                    Annotation(fee_raw=possible_fee, sentence=sentence)
                )

            self.annotations.append(predicted_annotations)

    def predict_spans(self, span_identifier: SpanIdentifier = None):
        """

        :return:
        """

        logging.debug(f"Predicting Spans")
        use_static = False

        if span_identifier is None:
            span_identifier = SpanIdentifier(self.cM)
            use_static = True

        num_sentences = range(len(self.sentences))

        for i in tqdm(num_sentences):
            for annotation in self.annotations[i]:

                p_role_positions = span_identifier.query(annotation, use_static)

                annotation.role_positions = p_role_positions
                annotation.roles = []

        logging.debug(f"Done predicting Spans")

    def get_annotations(self, sentence: List[str] = None):
        """

        :param sentence: The sentence to retrieve the annotations for.
        :return:
        """

        for i in len(self.sentences):

            if self.sentences[i] == sentence:
                return self.annotations[i]

        return None
