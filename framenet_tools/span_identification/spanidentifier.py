import logging
import os
import pickle

import spacy
import re
import torch

from torchtext import data
from tqdm import tqdm
from typing import List

from framenet_tools.config import ConfigManager
from framenet_tools.data_handler.annotation import Annotation
from framenet_tools.data_handler.reader import DataReader
from framenet_tools.utils.postagger import PosTagger
from framenet_tools.utils.static_utils import shuffle_concurrent_lists, pos_to_int
from framenet_tools.span_identification.spanidnetwork import SpanIdNetwork


def get_dataset(reader: DataReader):
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


class SpanIdentifier(object):
    """
    The Span Identifier for predicting possible role spans of a given sentence

    Includes multiple ways of predicting:
        -static
        -using allennlp
        -using a bilstm
    """

    def __init__(self, cM: ConfigManager):

        self.cM = cM
        self.network = None
        self.input_field = data.Field(
            dtype=torch.long, use_vocab=True, preprocessing=None
        )

        self.en_nlp = spacy.load("en_core_web_sm")
        self.dep_dict = []

    def query(
        self,
        embedded_sentence: List[float],
        annotation: Annotation,
        pos_tags: List[str],
        use_static: bool = True,
    ):
        """
        Predicts a possible span set for a given sentence.

        NOTE: This can be done static (only using syntax) or via an LSTM.

        :param pos_tags: The postags of the sentence
        :param embedded_sentence: The embedded words of the sentence
        :param annotation: The annotation of the sentence to predict
        :param use_static: True uses the syntactic static version, otherwise the NN
        :return: A list of possible span tuples
        """

        if use_static:
            return self.query_static(annotation)
        else:
            return self.query_nn(embedded_sentence, annotation, pos_tags)

    def query_nn(
        self, sentence: List[float], annotation: Annotation, pos_tags: List[str]
    ):
        """
        Predicts the possible spans using the LSTM.

        NOTE: In order to use this, the network must be trained beforehand

        :param pos_tags: The postags of the sentence
        :param sentence: The embedded words of the sentence
        :param annotation: The annotation of the sentence to predict
        :return: A list of possible span tuples
        """

        possible_roles = []
        count = 0
        new_span = -1

        doc = self.en_nlp(" ".join(sentence))

        combined = [
            [self.input_field.vocab.stoi[word]]
            + annotation.embedded_frame
            + [pos_to_int(pos_tag[1])]
            + [self.dep_to_int(token.dep_)]
            + [token.head.idx - token.idx]
            + [pos_to_int(token.head.tag_)]
            for word, pos_tag, token in zip(sentence, pos_tags, doc)
        ]

        bio_tags = self.network.predict(combined)[0]

        bio_tags = torch.argmax(bio_tags, 1)

        for bio_tag in bio_tags:

            if bio_tag == 0:
                new_span = count

            if bio_tag == 2 and new_span != -1:
                possible_roles.append((new_span, count - 1))
                new_span = -1

            count += 1

        return possible_roles

    def query_static(self, annotation: Annotation):
        """
        Predicts the set of possible spans just by the use of the static syntax tree.

        NOTE: deprecated!

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

        bio = ["O"] * sentence_length

        for role_position in annotation.role_positions:
            b = role_position[0]
            bio[b] = "B"

            for i in range(b + 1, role_position[1] + 1):
                bio[i] = "I"

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

    def get_dataset_comb(self, m_reader: DataReader):
        """
        Generates sentences with their BIO-tags

        :param m_reader: The DataReader to create the dataset from
        :return: A pair of concurrent lists containing the sequences and their labels
        """

        xs = []
        ys = []

        pos_tagger = PosTagger(self.cM.use_spacy)

        for annotations_sentence, sentence in zip(
            m_reader.annotations, m_reader.sentences
        ):

            pos_tags = pos_tagger.get_tags(sentence)

            doc = self.en_nlp(" ".join(sentence))

            for annotation in annotations_sentence:

                sent_len = len(sentence)
                spans = [2] * sent_len

                for role_pos in annotation.role_positions:

                    spans[role_pos[0]] = 0

                    for i in range(role_pos[0] + 1, role_pos[1] + 1):
                        spans[i] = 1

                ys.append(spans)

                combined = [
                    [self.input_field.vocab.stoi[word]]
                    + annotation.embedded_frame
                    + [pos_to_int(pos_tag[1])]
                    + [self.dep_to_int(token.dep_)]
                    + [token.head.idx - token.idx]
                    + [pos_to_int(token.head.tag_)]
                    for word, pos_tag, token in zip(sentence, pos_tags, doc)
                ]
                xs.append(combined)

        return xs, ys

    def dep_to_int(self, dep: str):
        """
        Converts a dependency feature into a number

        :param dep: The feature
        :return: A consistent number
        """

        if dep not in self.dep_dict:
            self.dep_dict.append(dep)

        return self.dep_dict.index(dep)

    def load(self):
        """
        Loads the saved model of the span identification network

        :return:
        """

        self.network = SpanIdNetwork(self.cM, 3)
        self.network.load_model("data/models/span_test.m")

    def save_model(self, name: str):
        """
        Saves a model as a file

        :param name: The path of the model to save to
        :return:
        """

        # Saving the current config
        self.cM.create_config(name + ".span.cfg")

        # Saving all Vocabs
        pickle.dump(self.input_field.vocab, open(name + ".span.in_voc", "wb"))

    def load_model(self, name: str):
        """
        Loads a model from a given file

        NOTE: This drops the current model!

        :param name: The path of the model to load
        :return:
        """

        # Loading config
        self.cM = ConfigManager(name + ".span.cfg")

        # Loading Vocabs
        in_voc = pickle.load(open(name + ".span.in_voc", "rb"))

        self.input_field.vocab = in_voc

        embed = torch.nn.Embedding.from_pretrained(self.input_field.vocab.vectors)

        self.network = SpanIdNetwork(self.cM, 3, embed)
        self.network.load_model("data/models/span_test.m")

    def gen_embedding_layer(self, reader: DataReader):
        """

        :param reader:
        :return:
        """

        input_field = data.Field(
            dtype=torch.long, use_vocab=True, preprocessing=None
        )  # , fix_length= max_length) #No padding necessary anymore, since avg
        output_field = data.Field(dtype=torch.long)
        data_fields = [("Sentence", input_field), ("Frame", output_field)]

        xs = []
        ys = []

        new_xs, new_ys = get_dataset(reader)
        xs += new_xs
        ys += new_ys

        shuffle_concurrent_lists([xs, ys])

        examples = [data.Example.fromlist([x, y], data_fields) for x, y in zip(xs, ys)]

        dataset = data.Dataset(examples, fields=data_fields)

        input_field.build_vocab(dataset)
        output_field.build_vocab(dataset)

        input_field.vocab.load_vectors("glove.6B.300d")

        embed = torch.nn.Embedding.from_pretrained(input_field.vocab.vectors)

        self.input_field = input_field

        return embed

    def train(self, mReader, mReaderDev):
        """
        Trains the model on all of the given annotations.

        :param annotations: A list of all annotations to train the model from
        :return:
        """

        embed = self.gen_embedding_layer(mReader)

        xs, ys = self.get_dataset_comb(mReader)

        dev_xs, dev_ys = self.get_dataset_comb(mReaderDev)

        shuffle_concurrent_lists([xs, ys])

        num_classes = 3

        self.network = SpanIdNetwork(self.cM, num_classes, embed)

        self.network.train_model(xs, ys, dev_xs, dev_ys)

    def predict_spans(self, m_reader: DataReader):
        """
        Predicts the spans of the currently loaded dataset.
        The predictions are saved in the annotations.

        NOTE: All loaded spans and roles are overwritten!

        :return:
        """

        logging.info(f"Predicting Spans")
        use_static = False

        # if span_identifier is None:
        #    span_identifier = SpanIdentifier(self.cM)
        #    use_static = True

        num_sentences = range(len(m_reader.sentences))

        for i in tqdm(num_sentences):
            for annotation in m_reader.annotations[i]:

                p_role_positions = self.query(
                    m_reader.sentences[i], annotation, m_reader.pos_tags[i], use_static
                )

                annotation.role_positions = p_role_positions
                annotation.roles = []

        logging.info(f"Done predicting Spans")
