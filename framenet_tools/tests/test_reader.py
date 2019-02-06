import pytest
import random
import string
import os

from framenet_tools.frame_identification.reader import DataReader, Annotation


def create_random_string(possible_chars: str, seq_length: int = 8):
    """
    Helper Function for generation of random strings.

    Inspired by: https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits-in-python

    :param seq_length:
    :param possible_chars:
    :return:
    """

    return "".join(random.choice(possible_chars) for _ in range(random.randint(1, seq_length)))


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
            # TODO Position in the sentence
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


def create_and_load(max_sentence_length: int):
    """

    :param max_sentence_length:
    :return:
    """

    num_sentences = random.randint(1, max_sentence_length)

    frames_file = create_frames_file(num_sentences, 10)
    sentences_file = create_sentences_file(num_sentences, 10)

    m_reader = DataReader()
    m_reader.read_data(sentences_file, frames_file)

    return m_reader, [sentences_file, frames_file]


def clean_up(files: list):
    """

    :param files:
    :return:
    """

    for file in files:
        os.remove(file)


def test_reader_simple():
    """
    A simple reader test

    NOTE: Reading in randomly (but well formatted) generated files.

    :return:
    """

    _, files = create_and_load(10)

    clean_up(files)


def test_reader_no_file():
    """
    Checks if Exception is raised in case of no specified file

    :return:
    """

    with pytest.raises(Exception):
        m_reader = DataReader()
        m_reader.read_data()


def test_reader_sizes():
    """
    A reader test checking if no line was lost during loading in either of both files

    NOTE: Reading in randomly (but well formatted) generated files.

    :return:
    """

    m_reader, files = create_and_load(10)
    handler = [m_reader.sentences, m_reader.annotations]

    for i in range(2):
        file = open(files[i])
        raw = file.read()
        file.close()

        raw = raw.rsplit("\n")
        # None empty lines counted
        line_count = sum(1 for line in raw if line != "")

        assert len(handler[i]) == line_count

    clean_up(files)


def test_sentences_correctness():
    """
    A reader test checking if any read sentences file was not loaded correctly.

    NOTE: Reading in randomly (but well formatted) generated files.

    :return:
    """

    m_reader, files = create_and_load(10)

    file = open(files[0])
    raw = file.read()
    file.close()

    raw = raw.rsplit("\n")

    for line, sentence in zip(raw, m_reader.sentences):

        line = [x for x in line.rsplit(" ") if x != ""]

        assert sentence == line

    clean_up(files)


def get_sentence(sentences: list, sentence_num: int):
    """
    Helper function converting raw sentences into a list of words.

    :param sentences: The list of sentences as strings
    :param sentence_num: The number of the sentence to get
    :return: The sentence as a list of words as strings
    """
    sentence = sentences[sentence_num]
    sentence = [word for word in sentence.rsplit(" ") if word != ""]

    return sentence


def test_frames_correctness():
    """
    A reader test checking if any read frames file was not loaded correctly.

    NOTE: Reading in randomly (but well formatted) generated files.

    :return:
    """

    m_reader, files = create_and_load(10)

    file = open(files[1])
    raw = file.read()
    file.close()

    file = open(files[0])
    sentences = file.read().rsplit("\n")
    file.close()

    raw = raw.rsplit("\n")
    reader_annotations = [x for y in m_reader.annotations for x in y]

    for line, annotation in zip(raw, reader_annotations):

        line = [x for x in line.rsplit("\t") if x != ""]

        orig_annotation = Annotation(line[3], line[4], line[5], line[6], get_sentence(sentences, int(line[7])))
        # annotation = Annotation(line[3], line[4], line[5], line[6], int(line[7]))
        assert annotation == orig_annotation

    clean_up(files)
