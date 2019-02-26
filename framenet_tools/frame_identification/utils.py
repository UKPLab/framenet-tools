import logging
import nltk
import random
import os
import py7zlib
import requests

from typing import List


required_resources = [
    ["taggers/", "averaged_perceptron_tagger"],
    ["tokenizers/", "punkt"],
    ["corpora/", "wordnet"],
    ["chunkers/", "maxent_ne_chunker"],
    ["corpora/", "words"]
]


def download_resources():
    """
    Checks if the required resources from nltk are installed, if not they are downloaded.

    :return:
    """

    logging.debug(f"Checking nltk resources.")
    for resource in required_resources:
        try:
            nltk.data.find(resource[0] + resource[1])
        except LookupError:
            logging.info(f"Did not find {resource[1]}, downloading...")
            nltk.download(resource[1])


def shuffle_concurrent_lists(l: List[List[object]]):
    """
    Shuffles multiple concurrent lists so that pairs of (x, y) from different lists are still at the same index.

    :param l: A list of concurrent lists
    :return: The list of shuffled concurrent lists
    """

    for x in l:
        random.seed(42)
        random.shuffle(x)

    return l


def extract7z(path: str):
    """
    Extracts 7z Archive

    :param path: The path of the archive
    :return:
    """

    with open(path, "rb") as in_file:
        arch = py7zlib.Archive7z(in_file)

        for content in arch.getmembers():

            with open(content.filename, "wb") as out_file:
                out_file.write(content.read())


def download_file(url: str, file_path: str):
    """
    Downloads a file and saves at a given path

    :param url: The URL of the file to download
    :param file_path: The destination of the file
    :return:
    """

    r = requests.get(url, stream=True)
    with open(file_path, "wb") as fd:
        logging.info(f"Downloading {r.url} and saving to {file_path}")
        for chunk in r.iter_content(chunk_size=10 * 1024 * 1024):
            fd.write(chunk)
