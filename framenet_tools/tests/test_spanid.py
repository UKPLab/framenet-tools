import pytest
import torch
import torch.nn as nn
import random

from typing import List

from framenet_tools.config import ConfigManager
from framenet_tools.data_handler.annotation import Annotation
from framenet_tools.role_identification.spanidnetwork import Net, SpanIdNetwork
from framenet_tools.role_identification.spanidentifier import SpanIdentifier
from framenet_tools.tests.test_reader import RandomFiles, create_random_string

N = 10


def create_random_sentence(n: int):
    """

    :param n:
    :return:
    """

    t = ""

    for i in range(n):
        t += create_random_string() + " "

    t = t.rsplit(" ")[:n]

    return t


def create_network(
    embedding_vocab_size: int = 2,
    embedding_dim: int = 2,
    num_classes: int = 2,
    cM: ConfigManager = ConfigManager(),
):
    """
    
    :param embedding_vocab_size:
    :param embedding_dim:
    :param num_classes:
    :param cM:
    :return:
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


def test_query_all_format():
    """
    Checks if the returned spans of the query_all-function are of the right type and well formatted.

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

    :return:
    """

    cM = ConfigManager()
    sI = SpanIdentifier(cM)

    test_str = create_random_sentence(n)

    predicted_spans = sI.query_all(Annotation(sentence=test_str))

    assert len(predicted_spans) == n * (n + 1) / 2


def test_query_static_format():
    """
    Checks if the returned spans of the static-function are of the right type and well formatted.

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

    :return:
    """

    span_id_network = create_network()

    assert isinstance(span_id_network, SpanIdNetwork)


def test_predict_nn():
    """

    :return:
    """

    n = 20
    num_classes = 5

    cM = ConfigManager()
    cM.use_cuda = False

    span_identifier_network = create_network(num_classes=num_classes, cM=cM)

    test_str = [random.randint(0, num_classes+1)] * n

    predicted_spans = span_identifier_network.predict(test_str)

    assert predicted_spans.shape == torch.Size([1, n-1, num_classes])


