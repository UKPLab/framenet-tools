import logging
import nltk
import random
import os
import py7zlib
import requests
import spacy

from typing import List
from subprocess import call


required_resources = [
    ["taggers/", "averaged_perceptron_tagger"],
    ["tokenizers/", "punkt"],
    ["corpora/", "wordnet"],
    ["chunkers/", "maxent_ne_chunker"],
    ["corpora/", "words"],
]


pos_dict = [
    "CC",
    "CD",
    "DT",
    "EX",
    "FW",
    "IN",
    "JJ",
    "JJR",
    "JJS",
    "LS",
    "MD",
    "NN",
    "NNS",
    "NNP",
    "NNPS",
    "PDT",
    "POS",
    "PRP",
    "PRP$",
    "RB",
    "RBR",
    "RBS",
    "RP",
    "SYM",
    "TO",
    "UH",
    "VB",
    "VBD",
    "VBG",
    "VBN",
    "VBP",
    "VBZ",
    "WDT",
    "WP",
    "WP$",
    "WRB",
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
        py7zlib.ArchiveFile

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


def extract_file(file_path: str):
    """
    Extracts a zipped file

    :param file_path: The file to extract
    :return:
    """

    call(["7z", "x", file_path])

    # TODO extract using python, NOTE ran into trouble because of 7z
    # raw = open(file_path,"rb")
    # archive = Archive7z(raw)
    # data = archive.getmember(archive.getnames()[0]).read()
    # raw.close()

    # Cleanup
    os.remove(file_path)


def get_spacy_en_model():
    """
    Installs the required en_core_web_sm model

    NOTE: Solution for Windows? TODO
    :return:
    """

    call(["python3", "-m", "spacy", "download", "en_core_web_sm"])


def download(url: str):
    """
    Downloads and extracts a file given as a url.

    NOTE: The paths should NOT be changed in order for pyfn to work
    NOTE: Only extracts 7z files

    :param url: The url from where to get the file
    :return:
    """

    # Simply adopt filename
    file_path = url.rsplit("/")[-1]

    logging.info(f"Downloading {file_path}")

    download_file(url, file_path)
    extract_file(file_path)


def get_sentences(raw: str, use_spacy: bool = False):
    """
    Parses a raw string of text into structured sentences.
    This is either done via nltk or spacy; default being nltk.

    :param raw: A raw string of text
    :param use_spacy: True to use spacy, otherwise nltk
    :return: A list of sentences, consisting of tokens
    """

    if use_spacy:
        return get_sentences_spacy(raw)

    return get_sentences_nltk(raw)


def get_sentences_spacy(raw: str):
    """
    The spacy version of the get_sentences method.

    :param raw: A raw string of text
    :return: A list of sentences, consisting of tokens
    """

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(raw)
    sents = [sent.string.strip() for sent in doc.sents]

    tokenizer = spacy.tokenizer.Tokenizer(nlp.vocab)

    sentences = []

    for sent in sents:
        tokens = nlp(sent)
        words = [token.text for token in tokens]
        sentences.append(words)

    return sentences


def get_sentences_nltk(raw: str):
    """
    The nltk version of the get_sentences method.

    :param raw: A raw string of text
    :return: A list of sentences, consisting of tokens
    """

    sents = nltk.sent_tokenize(raw)

    sentences = []

    for sent in sents:
        words = nltk.word_tokenize(sent)
        sentences.append(words)

    return sentences


def pos_to_int(pos: str):
    """
    Converts a pos tag to an integer according to the static dictionary.

    :param pos: The pos tag
    :return: The index of the pos tag
    """

    if pos in pos_dict:
        return pos_dict.index(pos)

    return -1
