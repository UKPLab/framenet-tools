import json
import logging
import numpy as np
import random

from tqdm import tqdm
from typing import List

from framenet_tools.config import ConfigManager
from framenet_tools.data_handler.annotation import Annotation
from framenet_tools.utils.postagger import PosTagger


class DataReader(object):
    """
    The top-level DataReader

    Stores all loaded data from every reader.
    """

    def __init__(self, cM: ConfigManager):

        self.cM = cM

        self.sentences = []
        self.annotations = []

        # Embedded
        self.embedded_sentences = []
        self.pos_tags = []

        # Flags
        self.is_annotated = None
        self.is_loaded = False

    def __eq__(self, x):
        """
        The overwriting of the comparison function

        :param x: Another instance of this class
        :return: True if equal, otherwise false
        """

        if not len(self.annotations) == len(x.annotations):
            return False

        equal = True

        for s_annos, d_annos in zip(self.annotations, x.annotations):
            for s_anno, d_anno in zip(s_annos, d_annos):
                if not s_anno == d_anno:
                    equal = False

        return equal

    def loaded(self, is_annotated: bool):
        """
        Helper for setting flags

        :param is_annotated: flag if loaded data was annotated
        :return:
        """

        self.is_loaded = True
        self.is_annotated = is_annotated

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

            if len(annotations) < 1:
                continue

            data_dict["sentence"] = annotations[0].sentence
            data_dict["sentence_id"] = sent_count
            data_dict["prediction"] = []

            sent_count += 1

            frame_count = 0

            for annotation in annotations:

                prediction_dict = dict()
                prediction_dict["id"] = frame_count
                prediction_dict["fee"] = annotation.fee_raw
                prediction_dict["frame"] = annotation.frame_confidence
                prediction_dict["position"] = annotation.position[0]
                prediction_dict["roles"] = []

                role_id = 0

                roles = annotation.roles

                if not annotation.roles:
                    roles = ["Default"] * len(annotation.role_positions)

                for span, role in zip(annotation.role_positions, roles):

                    span_dict = dict()
                    span_dict["role_id"] = role_id
                    span_dict["role"] = role
                    span_dict["span"] = span

                    role_id += 1

                    prediction_dict["roles"].append(span_dict)

                data_dict["prediction"].append(prediction_dict)
                frame_count += 1

            out_data.append(data_dict)

        if path is None:
            print(out_data)
            return

        with open(path, "w") as out:
            json.dump(out_data, out, indent=4)

    def import_from_json(self, path: str):
        """
        Reads the data from a given json file

        :param path: The path to the json file
        :return:
        """

        sent_num = 0

        with open(path) as file:
            json_data = json.load(file)

        for data_pair in json_data:

            self.sentences.append(data_pair["sentence"])

            prediction = data_pair["prediction"]

            for data in prediction:

                frame = None
                fee = None
                position = None

                if not data["frame"] == []:
                    confidence = [i[1] for i in data["frame"]]
                    confidence_max = np.asarray(confidence).argmax()

                    frame = data["frame"][confidence_max][0]

                if not data["fee"] == "":
                    fee = data["fee"]  # Frame evoking element

                if not data["position"] == "":
                    position = data["position"]
                    position = (position, position)

                role_positions = []
                roles = []

                for role_data in data["roles"]:
                    role_positions.append(tuple(role_data["span"]))
                    roles.append(role_data["role"])

                # As this original information is lost, simply equal fee and fee_raw
                fee_raw = fee

                if sent_num >= len(self.annotations):
                    self.annotations.append([])

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

            sent_num = sent_num + 1

    def embed_word(self, word: str):
        """
        Embeds a single word

        :param word: The word to embed
        :return: The vector of the embedding
        """

        embedded = self.cM.wEM.embed(word)

        if embedded is None:
            embedded = self.cM.wEM.embed(word.lower())

        if embedded is None:
            embedded = [random.random() / 10 for _ in range(300)]

        return embedded

    def embed_words(self, force: bool = False):
        """
        Embeds all words of all sentences that are currently saved in "sentences".

        NOTE: Can erase all previously embedded data!

        :param force: If true, all previously saved embeddings will be overwritten!
        :return:
        """

        # not not - meaning if the list is NOT empty!
        if not not self.embedded_sentences and not force:
            return

        self.cM.wEM.read_word_embeddings()

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
        Embeds a single frame.

        NOTE: if the embeddings of the frame can not be found, a random set of values is generated.

        :param frame: The frame to embed
        :return: The embedding of the frame
        """

        embedded = self.cM.fEM.embed(frame)

        if embedded is None:
            embedded = [random.random() / 6 for _ in range(100)]

        return embedded

    def embed_frames(self, force: bool = False):
        """
        Embeds all the sentences that are currently loaded.

        NOTE: if forced, overrides embedded data inside of the annotation objects

        :param force: If true, embeddings are generate even if they already exist
        :return:
        """

        if (not self.annotations[0][0].embedded_frame is None) and not force:
            return

        self.cM.fEM.read_frame_embeddings()

        logging.info("Embedding sentences")

        for annotations in tqdm(self.annotations):

            for annotation in annotations:

                annotation.embedded_frame = self.embed_frame(annotation.frame)

        logging.info("[Done] embedding sentences")

    def generate_pos_tags(self, force: bool = False):
        """
        Generates the POS-tags of all sentences that are currently saved.

        :param force: If true, the POS-tags will overwrite previously saved tags.
        :return:
        """

        # not not - meaning if the list is NOT empty!
        if not not self.pos_tags and not force:
            return

        pos_tagger = PosTagger(self.cM.use_spacy)
        count = 0

        self.pos_tags = []

        for sentence in self.sentences:
            tags = pos_tagger.get_tags(sentence)
            self.pos_tags.append(tags)

            if len(sentence) != len(tags):
                count += 1

    def get_annotations(self, sentence: List[str] = None):
        """
        Returns the annotation object for a given sentence.

        :param sentence: The sentence to retrieve the annotations for.
        :return: A annoation object
        """

        for i in len(self.sentences):

            if self.sentences[i] == sentence:
                return self.annotations[i]

        return None
