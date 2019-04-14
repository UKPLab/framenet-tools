import pytest
import torch
import torch.nn as nn
import random

from framenet_tools.config import ConfigManager
from framenet_tools.data_handler.annotation import Annotation
from framenet_tools.span_identification.spanidnetwork import SpanIdNetwork
from framenet_tools.span_identification.spanidentifier import SpanIdentifier
from framenet_tools.tests.test_reader import create_random_string

N = 10


def create_random_sentence(n: int):
    """
    Creates a random sentence with the length of n.

    NOTE: Randomized!

    :param n: The length of the sentence to generate
    :return: A list of random words
    """

    t = []

    for i in range(n):
        t.append(create_random_string())

    return t


def create_network(
    embedding_vocab_size: int = 2,
    embedding_dim: int = 2,
    num_classes: int = 2,
    cM: ConfigManager = ConfigManager(),
):
    """
    Creates a instance of SpanIdNetwork with the given parameters.

    :param embedding_vocab_size: The size of the embedding vocab
    :param embedding_dim: The dimension of the embeddings
    :param num_classes: The number of different classes
    :param cM: A instance of ConfigManager
    :return: A SpanIdNetwork object
    """

    cM.embedding_size = embedding_dim

    embedding_layer = nn.Embedding(embedding_vocab_size, embedding_dim)

    span_id_network = SpanIdNetwork(cM, embedding_layer, num_classes)

    return span_id_network


def test_span_identifier():
    """
    Simple test if a SpanIdentifier can be created.

    :return:
    """

    cM = ConfigManager()
    sI = SpanIdentifier(cM)

    assert isinstance(sI, SpanIdentifier)


@pytest.mark.parametrize("runs", range(N))
def test_query_all_format(runs: int):
    """
    Checks if the returned spans of the query_all-function are of the right type and well formatted.

    NOTE: Randomized!

    :return:
    """

    cM = ConfigManager()
    sI = SpanIdentifier(cM)

    test_str = create_random_sentence(10)

    predicted_spans = sI.query_all(Annotation(sentence=test_str))

    for span in predicted_spans:
        assert isinstance(span, tuple)
        assert span[0] <= span[1]


@pytest.mark.parametrize("n", range(N))
def test_query_all(n: int):
    """
    Tests if the size of the spans returned by query_all equals the expected size.

    NOTE: Randomized!

    :return:
    """

    cM = ConfigManager()
    sI = SpanIdentifier(cM)

    test_str = create_random_sentence(n)

    predicted_spans = sI.query_all(Annotation(sentence=test_str))

    assert len(predicted_spans) == n * (n + 1) / 2


@pytest.mark.parametrize("runs", range(N))
def test_query_static_format(runs: int):
    """
    Checks if the returned spans of the static-function are of the right type and well formatted.

    NOTE: Randomized!

    :return:
    """

    cM = ConfigManager()
    sI = SpanIdentifier(cM)

    test_str = create_random_sentence(10)

    predicted_spans = sI.query_static(Annotation(sentence=test_str))

    for span in predicted_spans:
        assert isinstance(span, tuple)
        assert span[0] <= span[1]


@pytest.mark.parametrize("n", range(1, N))
def test_query_static(n: int):
    """
    Tests the querying of the static syntax tree prediction.

    NOTE: Testing with random sentences can not be proven easily,
     therefore it simply checks if something was predicted.

    :return:
    """

    cM = ConfigManager()
    sI = SpanIdentifier(cM)

    test_str = ""

    for i in range(n):
        test_str += create_random_string() + " "

    test_str = test_str.rsplit(" ")[:n]

    predicted_spans = sI.query_static(Annotation(sentence=test_str))

    assert len(predicted_spans) > 0


def test_span_id_network():
    """
    Simple test if the SpanIdNetwork can be created.

    :return:
    """

    span_id_network = create_network()

    assert isinstance(span_id_network, SpanIdNetwork)


@pytest.mark.parametrize("runs", range(10))
def test_predict_nn(runs: int):
    """
    Tests whether the prediction format and sizes match the expected.

    NOTE: Randomized!

    :return:
    """

    n = 20
    num_classes = 2

    cM = ConfigManager()
    cM.use_cuda = False

    span_identifier_network = create_network(embedding_vocab_size=n, num_classes=num_classes, cM=cM)

    test_str = [random.randint(0, num_classes+1)] * n

    predicted_spans = span_identifier_network.predict(test_str)
    # print(predicted_spans)

    assert predicted_spans.shape == torch.Size([1, n-1, num_classes])


