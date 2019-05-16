import json
import logging
import random
import re

from allennlp.predictors.predictor import Predictor
from tqdm import tqdm
from typing import List

from framenet_tools.config import ConfigManager
from framenet_tools.data_handler.annotation import Annotation
from framenet_tools.utils.postagger import PosTagger


class DataReader(object):
    def __init__(
        self, cM: ConfigManager, path_sent: str = None, path_elements: str = None, raw_path: str = None
    ):

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


    def embed_word(self, word: str):
        """

        :param word:
        :return:
        """

        embedded = self.cM.wEM.embed(word)

        if embedded is None:
            embedded = self.cM.wEM.embed(word.lower())

        #if embedded is None:
        #    embedded = self.cM.wEM.embed(fee)

        #if embedded is None:
        #    embedded = self.cM.wEM.embed(fee.lower())

        if embedded is None:
            embedded = [random.random()/10 for _ in range(300)]

        return embedded

    def embed_words(self):
        """

        NOTE: erases previously embedded data
        :return:
        """

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

        :param frame:
        :return:
        """

        embedded = self.cM.fEM.embed(frame)

        if embedded is None:
            embedded = [random.random()/6 for _ in range(100)]

        return embedded


    def embed_frames(self):
        """

        NOTE: overrides embedded data inside of the annotation objects
        :return:
        """

        self.cM.fEM.read_frame_embeddings()

        logging.info("Embedding sentences")

        for annotations in tqdm(self.annotations):

            for annotation in annotations:

                annotation.embedded_frame = self.embed_frame(annotation.frame)

        logging.info("[Done] embedding sentences")

    def generate_pos_tags(self):
        """

        :return:
        """

        pos_tagger = PosTagger(self.cM.use_spacy)
        count = 0

        for sentence in self.sentences:
            tags = pos_tagger.get_tags(sentence)
            self.pos_tags.append(tags)

            if len(sentence) != len(tags):
                count += 1

        print(count)
        print(len(self.sentences))
        #exit()


    def get_annotations(self, sentence: List[str] = None):
        """

        :param sentence: The sentence to retrieve the annotations for.
        :return:
        """

        for i in len(self.sentences):

            if self.sentences[i] == sentence:
                return self.annotations[i]

        return None
