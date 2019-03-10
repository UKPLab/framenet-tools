import sys
import spacy

import en_core_web_sm
import torch
import torch.nn as nn

from copy import deepcopy
from torchtext import data
from typing import List

from framenet_tools.config import ConfigManager
from framenet_tools.data_handler.annotation import Annotation
#from framenet_tools.data_handler.reader import DataReader
from framenet_tools.frame_identification.utils import shuffle_concurrent_lists
from framenet_tools.role_identification.spanidnetwork import SpanIdNetwork


class SpanIdentifier(object):

    def __init__(self, cM: ConfigManager):

        # Create fields
        self.input_field = data.Field(
            dtype=torch.long, use_vocab=True, preprocessing=None
        )  # , fix_length= max_length) #No padding necessary anymore, since avg
        self.output_field = data.Field(dtype=torch.long)
        self.data_fields = [
            ("Sentence", self.input_field),
            ("BIO", self.output_field),
        ]

        self.cM = cM
        self.network = None

        self.nlp = en_core_web_sm.load()

    def query(self, annotiation: Annotation, use_static: bool = True):
        """

        :param annotiation:
        :param use_static:
        :return:
        """

        if use_static:
            return self.query_static(annotiation)
        else:
            return self.query_nn(annotiation)

    def query_nn(self, annotation: Annotation):
        """

        :param annotation:
        :return:
        """

        self.network.reset_hidden()

        possible_roles = []
        count = 0
        new_span = -1
        sent = []

        for word in annotation.sentence:

            sent.append(self.input_field.vocab.stoi[word])

        sent.append(self.input_field.vocab.stoi[annotation.fee_raw])

        bio_tags = self.network.predict(sent)[0]

        # print(bio_tags)
        bio_tags = torch.argmax(bio_tags, 1)
        # print(bio_tags)

        # exit()

        for bio_tag in bio_tags:
            # print(bio_tag)
            bio_tag = self.output_field.vocab.itos[bio_tag]
            # print(bio_tag)

            if bio_tag == 0:
                new_span = count

            if bio_tag == 2 and new_span != -1:
                possible_roles.append((new_span, count-1))
                new_span = -1

        return possible_roles

    def query_static(self, annotation: Annotation):
        """

        :param annotation:
        :return:
        """

        tokens = annotation.sentence

        possible_roles = []
        sentence = ""

        if len(tokens) > 0:
            sentence = tokens[0]

        for token in tokens:
            sentence += " " + token

        '''
        # Warning: Get way to large, way to fast...
        for i in range(len(sentence)):
            for j in range(i, len(sentence)):

                possible_roles.append((i, j))
        '''

        #sentence ="Autonomous cars shift insurance liability toward manufacturers"

        doc = self.nlp(sentence)

        '''
        for token in doc:
           
            min_index = sys.maxsize
            max_index = -1

            for child in token.children:

                position = list(doc).index(child)

                if position < min_index:
                    min_index = position

                if position > max_index:
                    max_index = position

            if max_index != -1: #and min != sys.maxsize:
                span = (min(min_index, token.i), max(max_index, token.i))
            else:
                span = ((token.i, token.i))

            possible_roles.append(span)


        #print(possible_roles)
        #exit()
        '''

        root = [token for token in doc if token.head == token][0]
        #print(root)

        combinations = self.traverse_syntax_tree(root)

        for combination in combinations:
            t = (min(combination), max(combination))
            if t not in possible_roles:
                possible_roles.append(t)

        #print(possible_roles)

        #exit()

        return possible_roles

    def traverse_syntax_tree(self, node: spacy.tokens.Token):
        spans = []
        retrieved_spans = []

        left_nodes = list(node.lefts)
        right_nodes = list(node.rights)

        for x in left_nodes:
            subs = self.traverse_syntax_tree(x)
            retrieved_spans += subs
            #spans.append([sub.append(node.i) for sub in subs])

        for x in right_nodes:
            subs = self.traverse_syntax_tree(x)
            retrieved_spans += subs
            #spans.append([sub.append(node.i) for sub in subs])

        # print(spans)
        # retrieved_spans = deepcopy(spans)
        for span in retrieved_spans:
            spans.append(span)
            spans.append(span + [node.i])


        if not spans:
            spans.append([node.i])

        #print(spans)
        return spans

    def evaluate(self, prediction, role_positions):
        """

        :param prediction:
        :param role_positions:
        :return:
        """

        return None

    def get_dataset(self, annotations: Annotation):
        """
        Loads the dataset and combines the necessary data

        :param file: A list of the two files to load
        :param predict_fees: A boolean whether to predict the frame evoking elements
        :return: xs: A list of senctences appended with its FEE
                ys: A list of frames corresponding to the given sentences
        """

        xs = []
        ys = []

        for annotation_sentences in annotations:
            for annotation in annotation_sentences:

                tags = self.generate_BIO_tags(annotation)

                # print(annotation.sentence)
                sentence = annotation.sentence + [annotation.fee_raw]
                # sentence = [[word, annotation.fee_raw] for word in annotation.sentence]
                # print(sentence)
                xs.append(sentence)
                ys.append(tags)

                '''
                for word, tag in zip(annotation.sentence, tags):
                    xs.append(word)
                    ys.append(tag)
                '''

        return xs, ys

    def generate_BIO_tags(self, annotation: Annotation):
        """

        :param annotation:
        :return:
        """

        sentence_length = len(annotation.sentence)

        bio = [2] * sentence_length

        for role_position in annotation.role_positions:
            b = role_position[0]
            bio[b] = 0

            for i in range(b, role_position[1]):
                bio[i] = 1

        return bio

    def to_one_hot(self, l: List[int]):
        """

        :param x:
        :return:
        """

        max_val = max(l)

        one_hots = [[0] * max_val] * len(l)

        for i in range(l):
            one_hots[i][l[i]] = 1

        return one_hots

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

    def train(self, sentences: List[List[str]], annotations: Annotation):
        """
        Trains the model on given files

        NOTE: If more than two file sets are given, they will be concatenated!

        :param files: A list of file sets
        :return:
        """

        xs, ys = self.get_dataset(annotations)

        # Not needed atm...
        # ys = self.to_one_hot(ys)

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

        #dev_iter = self.get_iter(self.cM.eval_files[0])

        self.input_field.vocab.load_vectors("glove.6B.300d")

        num_classes = len(self.output_field.vocab)

        embed = nn.Embedding.from_pretrained(self.input_field.vocab.vectors)

        self.network = SpanIdNetwork(self.cM, embed, num_classes)

        self.network.train_model(dataset_size, train_iter)

