import json
import logging
import random

from tqdm import tqdm
from typing import List

from framenet_tools.config import ConfigManager
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
                prediction_dict["roles"] = []

                role_id = 0

                for span in annotation.role_positions:

                    span_dict = dict()
                    span_dict["role_id"] = role_id
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

        if (not self.annotations[0].embedded_frame is None) and not force:
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
