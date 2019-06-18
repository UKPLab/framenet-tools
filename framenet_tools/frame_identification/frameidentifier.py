import json
from copy import deepcopy

import torch
import torch.nn as nn
from torchtext import data
import pickle
from typing import List

from framenet_tools.data_handler.annotation import Annotation
from framenet_tools.data_handler.reader import DataReader
from framenet_tools.fee_identification.feeidentifier import FeeIdentifier
from framenet_tools.frame_identification.frameidnetwork import FrameIDNetwork
from framenet_tools.config import ConfigManager
from framenet_tools.utils.static_utils import shuffle_concurrent_lists


class FrameIdentifier(object):
    """
    The FrameIdentifier

    Manages the neural network and dataset creation needed for training and evaluation.
    """

    def __init__(self, cM: ConfigManager):
        # Create fields
        self.input_field = data.Field(
            dtype=torch.long, use_vocab=True, preprocessing=None
        )  # , fix_length= max_length) #No padding necessary anymore, since avg
        self.output_field = data.Field(dtype=torch.long)
        self.data_fields = [
            ("Sentence", self.input_field),
            ("Frame", self.output_field),
        ]

        self.cM = cM
        self.network = None

    def get_dataset(self, reader: DataReader):
        """
        Loads the dataset and combines the necessary data

        :param reader: The reader that contains the dataset
        :return: xs: A list of sentences appended with its FEE
                ys: A list of frames corresponding to the given sentences
        """

        xs = []
        ys = []

        for annotation_sentences in reader.annotations:
            for annotation in annotation_sentences:
                xs.append([annotation.fee_raw] + annotation.sentence)
                ys.append(annotation.frame)

        return xs, ys

    def prepare_dataset(self, xs: List[str], ys: List[str], batch_size: int = None):
        """
        Prepares the dataset and returns a BucketIterator of the dataset

        :param batch_size: The batch_size to which the dataset will be prepared
        :param xs: A list of sentences
        :param ys: A list of frames corresponding to the given sentences
        :return: A BucketIterator of the dataset
        """

        if batch_size is None:
            batch_size = self.cM.batch_size

        examples = [
            data.Example.fromlist([x, y], self.data_fields) for x, y in zip(xs, ys)
        ]

        dataset = data.Dataset(examples, fields=self.data_fields)

        iterator = data.BucketIterator(dataset, batch_size=batch_size, shuffle=False)

        return iterator

    def evaluate(self, predictions: List[torch.tensor], xs: List[str], reader: DataReader):
        """
        Evaluates the model

        NOTE: for evaluation purposes use the function evaluate_file instead

        :param predictions: The predictions the model made on xs
        :param xs: The original fed in data
        :param reader: The reader from which xs was derived
        :return:
        """

        # Load correct answers for comparison:
        gold_xs, gold_ys = self.get_dataset(reader)

        tp = 0
        fp = 0
        fn = 0

        predictions = [i.item() for j in predictions for i in j]

        found = False

        for gold_x, gold_y in zip(gold_xs, gold_ys):
            for x, y in zip(xs, predictions):
                if gold_x == x and gold_y == self.output_field.vocab.itos[y]:
                    found = True
                    break

            if found:
                tp += 1
            else:
                fn += 1

            found = False

        for x, y in zip(xs, predictions):
            for gold_x, gold_y in zip(gold_xs, gold_ys):
                if gold_x == x and gold_y == self.output_field.vocab.itos[y]:
                    found = True

            if not found:
                fp += 1

            found = False

        return tp, fp, fn

    def query(self, annotation: Annotation):
        """

        :param annotation:
        :return:
        """

        x = [annotation.fee_raw] + annotation.sentence

        x = [[self.input_field.vocab.stoi[t]] for t in x]

        frame = self.network.query(x)
        frame = self.output_field.vocab.itos[frame.item()]

        return frame

    def write_predictions(self, file: str, out_file: str, fee_only: bool = False):
        """
        Prints the predictions of a given file

        :param file: The file to predict (either a raw file or annotated file set)
        :param out_file: The filename for saving the predictions
        :param fee_only: If True, only Frame Evoking Elements are predicted,
                         NOTE: In this case there is no need for either train or load a network
        :return:
        """

        if not fee_only and self.network is None:
            raise Exception("No network found! Train or load a network.")

        xs, ys = self.get_dataset([file], True)

        if not fee_only:
            dataset_iter = self.prepare_dataset(xs, ys, 1)
            predictions = self.network.predict(dataset_iter)
            prediction = iter(predictions)

        out_data = []
        sent_count = 0
        last_sentence = []

        for x in xs:
            if last_sentence != x[1:]:
                if not sent_count == 0:
                    out_data.append(data_dict)

                data_dict = dict()
                data_dict["sentence"] = x[1:]
                data_dict["sentence_id"] = sent_count
                data_dict["prediction"] = []
                last_sentence = x[1:]
                sent_count += 1
                frame_count = 0

            prediction_dict = dict()
            prediction_dict["id"] = frame_count
            prediction_dict["fee"] = x[0]
            if not fee_only:
                prediction_dict["frame"] = self.output_field.vocab.itos[
                    next(prediction).item()
                ]

            data_dict["prediction"].append(prediction_dict)

            frame_count += 1

        out_data.append(data_dict)

        with open(out_file, "w") as out:
            json.dump(out_data, out, indent=4)

    def save_model(self, name: str):
        """
        Saves a model as a file

        :param name: The path of the model to save to
        :return:
        """

        # Saving the current config
        self.cM.create_config(name + ".cfg")

        # Saving all Vocabs
        pickle.dump(self.output_field.vocab, open(name + ".out_voc", "wb"))
        pickle.dump(self.input_field.vocab, open(name + ".in_voc", "wb"))

        # Saving the actual network
        self.network.save_model(name + ".ph")

    def load_model(self, name: str):
        """
        Loads a model from a given file

        NOTE: This drops the current model!

        :param name: The path of the model to load
        :return:
        """

        # Loading config
        self.cM = ConfigManager(name + ".cfg")

        # Loading Vocabs
        out_voc = pickle.load(open(name + ".out_voc", "rb"))
        in_voc = pickle.load(open(name + ".in_voc", "rb"))

        self.output_field.vocab = out_voc
        self.input_field.vocab = in_voc

        num_classes = len(self.output_field.vocab)
        embed = nn.Embedding.from_pretrained(self.input_field.vocab.vectors)
        self.network = FrameIDNetwork(self.cM, embed, num_classes)

        self.network.load_model(name + ".ph")

    def evaluate_file(self, reader: DataReader, predict_fees: bool = False):
        """
        Evaluates the model on a given file set

        :param reader: The reader to evaluate on
        :return: A Triple of True Positives, False Positives and False Negatives
        """

        reader_copy = deepcopy(reader)

        if predict_fees:
            fee_finder = FeeIdentifier(self.cM)
            fee_finder.predict_fees(reader)

        xs, ys = self.get_dataset(reader)

        iter = self.prepare_dataset(xs, ys, 1)

        predictions = self.network.predict(iter)

        return self.evaluate(predictions, xs, reader_copy)

    def train(self, reader: DataReader, reader_dev: DataReader = None):
        """
        Trains the model on the given reader.

        NOTE: If no development reader is given, autostopping will be disabled!

        :param reader: The DataReader object which contains the training data
        :param reader_dev: The DataReader object for evaluation and auto stopping
        :return:
        """

        xs = []
        ys = []

        new_xs, new_ys = self.get_dataset(reader)
        xs += new_xs
        ys += new_ys

        shuffle_concurrent_lists([xs, ys])

        # Zip datasets and generate complete dictionary
        examples = [
            data.Example.fromlist([x, y], self.data_fields) for x, y in zip(xs, ys)
        ]

        dataset = data.Dataset(examples, fields=self.data_fields)

        self.input_field.build_vocab(dataset)
        self.output_field.build_vocab(dataset)

        dataset_size = len(xs)

        train_iter = self.prepare_dataset(xs, ys)

        dev_iter = self.get_iter(reader_dev)

        self.input_field.vocab.load_vectors("glove.6B.300d")

        num_classes = len(self.output_field.vocab)

        embed = nn.Embedding.from_pretrained(self.input_field.vocab.vectors)

        self.network = FrameIDNetwork(self.cM, embed, num_classes)

        self.network.train_model(dataset_size, train_iter, dev_iter)

    def get_iter(self, reader: DataReader):
        """
        Creates an Iterator for a given DataReader object.

        :param reader: The DataReader object
        :return: A Iterator of the dataset
        """

        xs, ys = self.get_dataset(reader)

        return self.prepare_dataset(xs, ys)
