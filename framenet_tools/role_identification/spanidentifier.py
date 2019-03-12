import spacy

import en_core_web_sm
import torch
import torch.nn as nn

from torchtext import data
from typing import List

from framenet_tools.config import ConfigManager
from framenet_tools.data_handler.annotation import Annotation
from framenet_tools.frame_identification.utils import shuffle_concurrent_lists
from framenet_tools.role_identification.spanidnetwork import SpanIdNetwork


class SpanIdentifier(object):
    def __init__(self, cM: ConfigManager):

        # Create fields
        self.input_field = data.Field(
            dtype=torch.long, use_vocab=True, preprocessing=None
        )
        self.output_field = data.Field(dtype=torch.long)
        self.data_fields = [("Sentence", self.input_field), ("BIO", self.output_field)]

        self.cM = cM
        self.network = None

        self.nlp = en_core_web_sm.load()

    def query(self, annotation: Annotation, use_static: bool = True):
        """
        Predicts a possible span set for a given sentence.

        NOTE: This can be done static (only using syntax) or via an LSTM.

        :param annotation: The annotation of the sentence to predict
        :param use_static: True uses the syntactic static version, otherwise the NN
        :return: A list of possible span tuples
        """

        if use_static:
            return self.query_static(annotation)
        else:
            return self.query_nn(annotation)

    def query_nn(self, annotation: Annotation):
        """
        Predicts the possible spans using the LSTM.

        NOTE: In order to use this, the network must be trained beforehand

        :param annotation: The annotation of the sentence to predict
        :return: A list of possible span tuples
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

        bio_tags = torch.argmax(bio_tags, 1)

        for bio_tag in bio_tags:
            bio_tag = self.output_field.vocab.itos[bio_tag]

            if bio_tag == 0:
                new_span = count

            if bio_tag == 2 and new_span != -1:
                possible_roles.append((new_span, count - 1))
                new_span = -1

        return possible_roles

    def query_static(self, annotation: Annotation):
        """
        Predicts the set of possible spans just by the use of the static syntax tree.

        :param annotation: The annotation of the sentence to predict
        :return: A list of possible span tuples
        """

        tokens = annotation.sentence

        possible_roles = []
        sentence = ""

        if len(tokens) > 0:
            sentence = tokens[0]

        for token in tokens:
            sentence += " " + token

        doc = self.nlp(sentence)

        """
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
        """

        root = [token for token in doc if token.head == token][0]

        combinations = self.traverse_syntax_tree(root)

        for combination in combinations:
            t = (min(combination), max(combination))
            if t not in possible_roles:
                possible_roles.append(t)

        return possible_roles

    def query_all(self, annotation: Annotation):
        """
        Returns all possible spans of a sentence.
        Therefore all correct spans are predicted, achieving a perfect Recall score, but close to 0 in Precision.

        NOTE: This creates a power set! Meaning there will be 2^N elements returned (N: words in senctence).

        :param annotation: The annotation of the sentence to predict
        :return: A list of ALL possible span tuples
        """

        possible_roles = []
        sentence = annotation.sentence

        # Warning: Gets way to large, way to fast...
        for i in range(len(sentence)):
            for j in range(i, len(sentence)):
                possible_roles.append((i, j))

        return possible_roles

    def traverse_syntax_tree(self, node: spacy.tokens.Token):
        """
        Traverses a list, starting from a given node and returns all spans of all its subtrees.

        NOTE: Recursive

        :param node: The node to start from
        :return: A list of spans of all subtrees
        """
        spans = []
        retrieved_spans = []

        left_nodes = list(node.lefts)
        right_nodes = list(node.rights)

        for x in left_nodes:
            subs = self.traverse_syntax_tree(x)
            retrieved_spans += subs

        for x in right_nodes:
            subs = self.traverse_syntax_tree(x)
            retrieved_spans += subs

        for span in retrieved_spans:
            spans.append(span)
            spans.append(span + [node.i])

        if not spans:
            spans.append([node.i])

        return spans

    def get_dataset(self, annotations: List[List[Annotation]]):
        """
        Loads the dataset and combines the necessary data

        :param annotations: A List of all annotations containing all sentences
        :return: xs: A list of senctences appended with its FEE
                 ys: A list of frames corresponding to the given sentences
        """

        xs = []
        ys = []

        for annotation_sentences in annotations:
            for annotation in annotation_sentences:

                tags = self.generate_BIO_tags(annotation)

                sentence = annotation.sentence + [annotation.fee_raw]
                xs.append(sentence)
                ys.append(tags)

                """
                for word, tag in zip(annotation.sentence, tags):
                    xs.append(word)
                    ys.append(tag)
                """

        return xs, ys

    def generate_BIO_tags(self, annotation: Annotation):
        """
        Generates a list of (B)egin-, (I)nside-, (O)utside- tags for a given annotation.

        :param annotation: The annotation to convert
        :return: A list of BIO-tags
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
        Helper Function that converts a list of numerals into a list of one-hot encoded vectors

        :param l: The list to convert
        :return: A list of one-hot vectors
        """

        max_val = max(l)

        one_hots = [[0] * max_val] * len(l)

        for i in range(len(l)):
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

    def train(self, annotations: List[List[Annotation]]):
        """
        Trains the model on all of the given annotations.

        :param annotations: A list of all annotations to train the model from
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

        # dev_iter = self.get_iter(self.cM.eval_files[0])

        self.input_field.vocab.load_vectors("glove.6B.300d")

        num_classes = len(self.output_field.vocab)

        embed = nn.Embedding.from_pretrained(self.input_field.vocab.vectors)

        self.network = SpanIdNetwork(self.cM, embed, num_classes)

        self.network.train_model(dataset_size, train_iter)
