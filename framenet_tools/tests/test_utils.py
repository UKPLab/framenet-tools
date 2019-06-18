import logging
import os
import pytest

from typing import List

from framenet_tools.evaluator import calc_f
from framenet_tools.utils.static_utils import (
    shuffle_concurrent_lists,
    extract7z,
    download_file,
    get_sentences,
)


@pytest.mark.parametrize(
    "testdata, expected",
    [([0] * 3, [0] * 3), ([1, 0, 0], [1, 1, 1]), ([1, 1, 1], [0.5, 0.5, 0.5])],
)
def test_calc_f(testdata: List[int], expected: List[float]):
    """
    Tests the calculation of Precision, Recall and F1-Score

    :param testdata: A list including the count of True Positives, False Positives and False Negatives
    :param expected: A list of the expected Precision, Recall and F1-Score
    :return:
    """

    pr, re, f = calc_f(testdata[0], testdata[1], testdata[2])

    assert pr == expected[0]
    assert re == expected[1]
    assert f == expected[2]


@pytest.mark.parametrize(
    "testdata, expected",
    [
        ([1, 2, 3, 4, 5], [6, 7, 8, 9, 0]),
        ([1, 0, 0, 5, 2, 45, 63], [1, 1, 4, 12, 97, 4, 1]),
        ([1, 4, 4, 1, 1, 2], [12, 13, 7, 0.5, 0.5, 9]),
    ],
)
def test_shuffle(testdata: List[object], expected: List[object]):
    """
    Test the correctness of the shuffling function in utils

    NOTE: Each shuffling is required to keep concurrency!

    :param testdata: A list of objects
    :param expected: A list of objects
    :return:
    """

    testdata2 = testdata.copy()
    expected2 = expected.copy()

    shuffle_concurrent_lists([testdata, expected])

    for i in range(len(testdata)):
        x = testdata[i]
        y = expected[i]

        assert (x, y) in [(s, t) for s, t in zip(testdata2, expected2)]

# TODO Find solution
'''
def test_extraction():

    testdir = "data/testing/"

    # Setup test dir
    if not os.path.exists(testdir):
        os.makedirs(testdir)

    url = "https://github.com/akb89/pyfn/releases/download/v1.0.0/lib.7z"
    file_path = testdir + url.rsplit("/")[-1]

    logging.info(f"Downloading {file_path}")

    if not os.path.exists(file_path):
        download_file(url, file_path)

    extract7z(file_path)
'''

@pytest.mark.parametrize(
    "use_spacy",
    [
        True,
        False
    ],
)
def test_tokenization(use_spacy: bool):

    text = "This is some random text, that should be splitted correctly. This is another sentence. And a third one."
    sentences = [
        [
            "This",
            "is",
            "some",
            "random",
            "text",
            ",",
            "that",
            "should",
            "be",
            "splitted",
            "correctly",
            ".",
        ],
        ["This", "is", "another", "sentence", "."],
        ["And", "a", "third", "one", "."],
    ]

    tokenized = get_sentences(text, use_spacy)

    assert tokenized == sentences
