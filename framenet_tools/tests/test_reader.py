import pytest
import random
import string
import os

from framenet_tools.frame_identification.reader import DataReader


def create_random_string(possible_chars: str, seq_length: int = 8):
    """
    Helper Function for generation of random strings.

    Inspired by: https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits-in-python

    :param seq_length:
    :param possible_chars:
    :return:
    """

    return "".join(random.choice(possible_chars) for _ in range(random.randint(1, seq_length)))


def test_string():
    print(create_random_string(string.ascii_lowercase, 10))

    num_sentences = random.randint(1, 10)

    frames_file = create_frames_file(num_sentences, 10)
    sentences_file = create_sentences_file(num_sentences, 10)

    m_reader = DataReader()
    m_reader.read_data(sentences_file, frames_file)

    os.remove(sentences_file)
    os.remove(frames_file)


def create_frames_file(sentence_count: int, max_frame_count: int):
    """

    :param sentence_count:
    :param max_frame_count:
    :return:
    """

    file_name = create_random_string(string.ascii_lowercase, 8)
    content = ""

    # Sentences
    for i in range(sentence_count):
        # Frames per sentence
        for j in range(random.randint(1, max_frame_count)):

            # Static part
            content += "1\t0.0\t" + str(random.randint(0, 5)) + "\t"
            # Frame
            content += create_random_string(string.ascii_letters + string.digits, 20) + "\t"
            # FEE
            content += create_random_string(string.ascii_letters + string.digits, 20) + "\t"
            # TODO Postion in the sentence
            content += "5\t"
            # Raw FEE
            content += create_random_string(string.ascii_letters + string.digits, 20) + "\t"
            # Sentence number (starting at 0) and new line
            content += str(i) + "\n"

    file = open(file_name, "w")
    file.write(content)
    file.close()

    return file_name


def create_sentences_file(sentence_count: int, max_word_count: int):
    """
    Helper Function for generation of random ".sentences" files.

    NOTE: Randomized!

    :param sentence_count: The max possible amount of lines/sentences in the file
    :param max_word_count: The max possible amount of words in a line/sentence
    :return: The name of the generated file
    """

    file_name = create_random_string(string.ascii_lowercase, 8)
    content = ""

    for i in range(sentence_count):
        for j in range(random.randint(1, max_word_count)):
            content += create_random_string(string.ascii_letters + string.digits) + " "

        content += "\n"

    file = open(file_name, "w")
    file.write(content)
    file.close()

    return file_name



def test_reader_simple():
    m_reader = DataReader()
    m_reader.read_data()


def test_reader_no_file():
    """
    Checks if Exception is raised in case of no specified file
    :return:
    """

    with pytest.raises(Exception):
        m_reader = DataReader()
        m_reader.read_data()
