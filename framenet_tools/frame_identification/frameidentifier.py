import json
import torch
import torch.nn as nn
from torchtext import data
import pickle
from typing import List

from framenet_tools.frame_identification.reader import DataReader
from framenet_tools.frame_identification.frameidnetwork import FrameIDNetwork
from framenet_tools.config import ConfigManager
from framenet_tools.frame_identification.utils import shuffle_concurrent_lists


class FrameIdentifier(object):
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

    def get_dataset(self, file: List[str], predict_fees: bool):
        """
        Loads the dataset and combines the necessary data

        :param file: A list of the two files to load
        :param predict_fees: A boolean whether to predict the frame evoking elements
        :return: xs: A list of senctences appended with its FEE
                ys: A list of frames corresponding to the given sentences
        """

        reader = DataReader()
        if len(file) == 2:
            reader.read_data(file[0], file[1])
        else:
            reader.read_raw_text(file[0])

        if predict_fees:
            reader.predict_fees()

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

    def evaluate(self, predictions: List[torch.tensor], xs: List[str], file: List[str]):
        """
        Evaluates the model

        NOTE: for evaluation purposes use the function evaluate_file instead

        :param predictions: The predictions the model made on xs
        :param xs: The original fed in data
        :param file: The file from which xs was derived
        :return:
        """

        # Load correct answers for comparison:
        gold_xs, gold_ys = self.get_dataset(file, False)

        tp = 0
        fp = 0
        fn = 0

        predictions = [i.item() for j in predictions for i in j]
        print(len(predictions))
        print(predictions)
        print(len(xs))
        # print(len(ys))

        # dataset = reformat_dataset(predictions, xs, ys)
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
        self.cM.load_config(name + ".cfg")

        # Loading Vocabs
        out_voc = pickle.load(open(name + ".out_voc", "rb"))
        in_voc = pickle.load(open(name + ".in_voc", "rb"))

        self.output_field.vocab = out_voc
        self.input_field.vocab = in_voc

        num_classes = len(self.output_field.vocab)
        embed = nn.Embedding.from_pretrained(self.input_field.vocab.vectors)
        self.network = FrameIDNetwork(self.cM, embed, num_classes)

        self.network.load_model(name + ".ph")

    def evaluate_file(self, file: List[str]):
        """
        Evaluates the model on a given file set

        :param file: The files to evaluate on
        :return: A Triple of True Positives, False Positives and False Negatives
        """

        xs, ys = self.get_dataset(file, False)

        iter = self.prepare_dataset(xs, ys, 1)

        predictions = self.network.predict(iter)

        return self.evaluate(predictions, xs, file)

    def train(self, files: List[str]):
        """
        Trains the model on given files

        NOTE: If more than two file sets are given, they will be concatenated!

        :param files: A list of file sets
        :return:
        """

        xs = []
        ys = []

        for file in files:
            new_xs, new_ys = self.get_dataset(file, False)
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

        dev_iter = self.get_iter(self.cM.eval_files[0])

        self.input_field.vocab.load_vectors("glove.6B.300d")

        num_classes = len(self.output_field.vocab)

        embed = nn.Embedding.from_pretrained(self.input_field.vocab.vectors)

        self.network = FrameIDNetwork(self.cM, embed, num_classes)

        self.network.train_model(train_iter, dev_iter)

    def get_iter(self, file: str):
        """

        :param file:
        :return:
        """

        xs, ys = self.get_dataset(file, False)

        return self.prepare_dataset(xs, ys)
