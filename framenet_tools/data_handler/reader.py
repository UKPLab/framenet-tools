import json
import logging
import re

import nltk
import spacy
from allennlp.predictors.predictor import Predictor
from tqdm import tqdm
from typing import List

from framenet_tools.fee_identification.feeidentifier import FeeIdentifier
from framenet_tools.frame_identification.utils import download_resources, get_sentences, get_spacy_en_model
from framenet_tools.config import ConfigManager
from framenet_tools.span_identification.spanidentifier import SpanIdentifier
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

        # Embedded
        self.embedded_sentences = []

        # Flags
        self.is_annotated = None
        self.is_loaded = False

        self.dataset = []

        download_resources()
        get_spacy_en_model()

        self.nlp = spacy.load("en_core_web_sm")

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
            fee_raw = element_data[6].rsplit(" ")[0]  # Frame evoking element as it appeared

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
            # print("Removed empty line at eof")
            elements = elements[: len(elements) - 1]

        if sentences[len(sentences) - 1] == "":
            # print("Removed empty line at eof")
            sentences = sentences[: len(sentences) - 1]

        # print(sentences)

        self.digest_raw_data(elements, sentences)

        self.loaded(True)

    def export_to_json(self, path: str):
        """
        Exports the list of annotations to a json file

        :param path: The path of the json file
        :return:
        """

        out_data = []
        sent_count = 0

        for annotations in self.annotations:
            data_dict = dict()
            data_dict["sentence"] = annotations[0].sentence
            data_dict["sentence_id"] = sent_count
            data_dict["prediction"] = []

            sent_count += 1

            frame_count = 0

            for annotation in annotations:

                prediction_dict = dict()
                prediction_dict["id"] = frame_count
                prediction_dict["fee"] = annotation.fee_raw
                prediction_dict["frame"] = annotation.frame

                data_dict["prediction"].append(prediction_dict)
                frame_count += 1

            out_data.append(data_dict)

        with open(path, "w") as out:
            json.dump(out_data, out, indent=4)

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

    def pred_allen(self):
        """

        :return:
        """

        print("starting")

        predictor = Predictor.from_path(
            "https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz")

        num_sentences = range(len(self.sentences))

        for i in tqdm(num_sentences):

            sentence = " ".join(self.sentences[i])

            prediction = predictor.predict(sentence)

            verbs = [t["verb"] for t in prediction["verbs"]]

            for annotation in self.annotations[i]:

                spans = []

                if annotation.fee_raw in verbs:
                    #print("d")
                    desc = prediction["verbs"][verbs.index(annotation.fee_raw)]["description"]

                    c = 0


                    while re.search("\[ARG[" + str(c) + "]: [^\]]*", desc) is not None:

                        span = re.search("\[ARG[" + str(c) + "]: [^\]]*", desc).span()

                        arg = desc[span[0]+7:span[1]]

                        # arg = nltk.word_tokenize(arg)
                        arg = self.nlp(arg)

                        for j in range(len(annotation.sentence)):

                            word = annotation.sentence[j]

                            if word == arg[0].text:
                                saved = j

                                for arg_word in arg:

                                    if not arg_word.text == annotation.sentence[j]:
                                        break

                                    saved2 = j
                                    j+=1


                        #annotation.sentence.index()

                        spans.append((saved, saved2))

                        c += 1

                annotation.role_positions = spans


                # print(prediction["verbs"])


    def predict_spans(self, span_identifier: SpanIdentifier = None):
        """
        Predicts the spans of the currently loaded dataset.
        The predictions are saved in the annotations.

        NOTE: All loaded spans and roles are overwritten!

        :return:
        """

        self.pred_allen()

        return

        logging.info(f"Predicting Spans")
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

        logging.info(f"Done predicting Spans")

    def embed_word(self, word: str):
        """

        :param word:
        :return:
        """

        embedded = self.cM.wEM.embed(word)

        return embedded

    def embed_words(self):
        """

        NOTE: erases previously embedded data
        :return:
        """

        self.embedded_sentences = []

        logging.info("Embedding sentences")

        for sentence in tqdm(self.sentences):
            embedded_sentence = []

            for word in sentence:
                embedded_sentence.append(self.embed_word(word))

            self.embedded_sentences.append(embedded_sentence)


        logging.info("[Done] embedding sentences")

    def embed_frame(self, frame: str):
        """

        :param frame:
        :return:
        """

        embedded = self.cM.fEM.embed(frame)

        return embedded


    def embed_frames(self):
        """

        NOTE: overrides embedded data inside of the annotation objects
        :return:
        """

        logging.info("Embedding sentences")

        for annotations in tqdm(self.annotations):

            for annotation in annotations:

                annotation.embedded_frame = self.embed_frame(annotation.frame)

        logging.info("[Done] embedding sentences")

    def get_annotations(self, sentence: List[str] = None):
        """

        :param sentence: The sentence to retrieve the annotations for.
        :return:
        """

        for i in len(self.sentences):

            if self.sentences[i] == sentence:
                return self.annotations[i]

        return None
