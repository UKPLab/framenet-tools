import logging
import nltk

from typing import List

# Static definitions
from framenet_tools.config import ConfigManager
from framenet_tools.data_handler.annotation import Annotation
from framenet_tools.data_handler.reader import DataReader
from framenet_tools.utils.postagger import PosTagger

punctuation = (".", ",", ";", ":", "!", "?", "/", "(", ")", "'")  # added
forbidden_words = (
    "a",
    "an",
    "as",
    "for",
    "i",
    "in particular",
    "it",
    "of course",
    "so",
    "the",
    "with",
)
preceding_words_of = (
    "%",
    "all",
    "face",
    "few",
    "half",
    "majority",
    "many",
    "member",
    "minority",
    "more",
    "most",
    "much",
    "none",
    "one",
    "only",
    "part",
    "proportion",
    "quarter",
    "share",
    "some",
    "third",
)
following_words_of = ("all", "group", "their", "them", "us")
loc_preps = (
    "above",
    "against",
    "at",
    "below",
    "beside",
    "by",
    "in",
    "on",
    "over",
    "under",
)
temporal_preps = ("after", "before")
dir_preps = ("into", "through", "to")
forbidden_pos_prefixes = (
    "PR",
    "CC",
    "IN",
    "TO",
    "PO",
)  # added "PO": POS = genitive marker
direct_object_labels = ("OBJ", "DOBJ")  # accomodates MST labels and Standford labels


def should_include_token(p_data: list):
    """
    A static syntactical prediction of possible Frame Evoking Elements

    :param p_data: A list of lists containing token, pos_tag, lemma and NE
    :return: A list of possible FEEs
    """

    num_tokens = len(p_data)
    targets = []
    no_targets = []
    for idx in range(num_tokens):
        token = p_data[idx][0].lower().strip()
        pos = p_data[idx][1].strip()
        lemma = p_data[idx][2].strip()
        ne = p_data[idx][3].strip()
        if idx >= 1:
            precedingWord = p_data[idx - 1][0].lower().strip()
            preceding_pos = p_data[idx - 1][1].strip()
            preceding_lemma = p_data[idx - 1][2].strip()
            preceding_ne = p_data[idx - 1][3]
        if idx < num_tokens - 1:
            following_word = p_data[idx + 1][0].lower()
            following_pos = p_data[idx + 1][1]
            following_ne = p_data[idx + 1][3]
        if (
            token in forbidden_words
            or token in loc_preps
            or token in dir_preps
            or token in temporal_preps
            or token in punctuation
            or pos[: min(2, len(pos))] in forbidden_pos_prefixes
            or token == "course"
            and precedingWord == "of"
            or token == "particular"
            and precedingWord == "in"
        ):
            no_targets.append(token)
        elif token == "of":
            if (
                preceding_lemma in preceding_words_of
                or following_word in following_words_of
                or preceding_pos.startswith("JJ")
                or preceding_pos.startswith("CD")
                or following_pos.startswith("CD")
            ):
                targets.append(token)
            if following_pos.startswith("DT"):
                if idx < num_tokens - 2:
                    following_following_pos = p_data[idx + 2][1]
                    if following_following_pos.startswith("CD"):
                        targets.append(token)
            if (
                following_ne.startswith("GPE")
                or following_ne.startswith("LOCATION")
                or preceding_ne.startswith("CARDINAL")
            ):
                targets.append(token)
        elif token == "will":
            if pos == "MD":
                no_targets.append(token)
            else:
                targets.append(token)
        elif lemma == "be":
            no_targets.append(token)
        else:
            targets.append(token)
    # print("targets: " + str(targets))
    # print("NO targets:\n" + str(notargets))
    return targets


class FeeIdentifier(object):
    def __init__(self, cM: ConfigManager):

        self.cM = cM

    def identify_targets(self, sentence: list):
        """
        Identifies targets for a given sentence

        :param sentence: A list of words in a sentence
        :return: A list of targets
        """

        tokens = nltk.word_tokenize(sentence)
        pos_tagger = PosTagger(self.cM.use_spacy)
        p_data = pos_tagger.get_tags(tokens)
        targets = should_include_token(p_data)

        return targets

    """
    def load_dataset(self, file):
        reader = Data_reader(file[0], file[1])
        reader.read_data()
        dataset = reader.get_dataset()

        return self.sum_FEEs(dataset)
    """

    def query(self, x: List[str]):
        """
        Query a prediction of FEEs for a given sentence

        :param x: A list of words in a sentence
        :return: A list of predicted FEEs
        """

        tokens = x[0]

        pData = self.get_tags(tokens)

        possible_fees = should_include_token(pData)

        return possible_fees

    def predict_fees(self, dataset: List[List[str]]):
        """
        Predicts all FEEs for a complete datset

        :param dataset: The dataset to predict
        :return: A list of predictions
        """
        predictions = []

        for data in dataset:
            prediction = self.query(data)
            predictions.append(prediction)

        return predictions

    def evaluate_acc(self, dataset: List[List[str]]):
        """
        Evaluates the accuracy of the Frame Evoking Element Identifier

        NOTE: F1-Score is a better way to evaluate the Identifier, because it tends to predict too many FEEs

        :param dataset: The dataset to evaluate
        :return: A Triple of the count of correct elements, total elements and the accuracy
        """
        correct = 0
        total = 0

        for data in dataset:
            predictions = self.query(data)

            total += len(data[1])

            for prediction in predictions:
                if prediction in data[1]:
                    correct += 1

        acc = correct / total

        return correct, total, acc

    def predict_fees(self, mReader: DataReader):
        """
        Predicts the Frame Evoking Elements
        NOTE: This drops current annotation data

        :return:
        """

        mReader.annotations = []
        #fee_finder = FeeIdentifier(self.cM)

        for sentence in mReader.sentences:
            possible_fees = self.query([sentence])
            predicted_annotations = []

            # Create new Annotation for each possible frame evoking element
            for possible_fee in possible_fees:
                predicted_annotations.append(
                    Annotation(fee_raw=possible_fee, sentence=sentence)
                )

            self.annotations.append(predicted_annotations)
