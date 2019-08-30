import os
import random
import shutil
import string

import pytest
from framenet_tools.config import ConfigManager
from framenet_tools.data_handler.semaforreader import SemaforReader
from framenet_tools.data_handler.semevalreader import SemevalReader
from framenet_tools.main import eval_args, create_argparser
from tests.test_reader import RandomFiles, create_random_string


# Parameter to run tests N times.
N = 4

# Simple mode
simple_mode = True

# Adjust path if necessary, NOTE: not needed for dummy testing.
data_path = "../"


"""
WARNING: The following tests check complete functionality of the system!

Therefore running a variety of complete training and evaluation tasks.
This can take a long time!

Also note that each test evaluates exactly one argset instead of programmatically 
do it all at once, for better overview and debugging.

To reduce the system load and duration, the simple mode can be activated. Note that this 
only tests the system on self generated dummy files.
"""


def adjust_config(simple_mode: bool):
    """
    Helper function for adjusting the config to match the testing purposes
    duration.

    NOTE: Simple mode tests everything on a single sentence-dummy-file!

    :param simple_mode: Specify whether simple mode should be used
    :return:
    """

    if simple_mode:

        shutil.copy("semeval_dummy.xml", "train.semeval_dummy.xml")
        shutil.copy("semeval_dummy.xml", "dev.semeval_dummy.xml")
        shutil.copy("semeval_dummy.xml", "test.semeval_dummy.xml")

        shutil.copy("semafor_dummy.sentences", "train.sentences")
        shutil.copy("semafor_dummy.frame.elements", "train.frame.elements")

        cM.semeval_train = ["train.semeval_dummy.xml"]
        cM.semeval_dev = ["dev.semeval_dummy.xml"]
        cM.semeval_test = ["test.semeval_dummy.xml"]

        cM.train_files = [["train.sentences", "train.frame.elements"]]

    else:

        cM.semeval_train = [data_path + cM.semeval_train[0]]
        cM.semeval_dev = [data_path + cM.semeval_dev[0]]
        cM.semeval_test = [data_path + cM.semeval_test[0]]

        train_files = []

        for files in cM.train_files:
            t = []

            for file in files:
                t.append(data_path + file)

            train_files.append(t)

        cM.train_files = train_files

    cM.num_epochs = 1

    cM.semeval_all = cM.semeval_train + cM.semeval_dev + cM.semeval_test

    cM.create_config("config.file")


os.remove("config.file")

cM = ConfigManager("config.file")

adjust_config(simple_mode)


def test_train_feeid():
    """
    Test for training the frame evoking element identification

    NOTE: This stage can not be trained, though for completeness it needs to be tested.

    :return:
    """

    eval_args(create_argparser(), ["train", "--feeid"])


def test_train_frameid():
    """
    Test for training the frame identification

    NOTE: Might take a long time!

    :return:
    """

    eval_args(create_argparser(), ["train", "--frameid"])


def test_train():
    """
    Test for training the complete pipeline

    NOTE: Might take a long time!

    :return:
    """

    eval_args(create_argparser(), ["train"])


def test_train_frameid_batchsize():
    """
    Test for training the frame identification with a manually set batchsize.

    NOTE: Might take a long time!
    NOTE: Randomized!

    :return:
    """

    batch_size = random.randint(50, 500)

    eval_args(
        create_argparser(), ["train", "--frameid", "--batchsize", str(batch_size)]
    )


def test_eval_feeid():
    """
    Test for evaluating the frame evoking element identification.

    :return:
    """

    eval_args(create_argparser(), ["evaluate", "--feeid"])


def test_eval_frameid():
    """
    Test the evaluation of the frame identification.

    :return:
    """

    eval_args(create_argparser(), ["evaluate", "--frameid"])


def test_eval_all():
    """
    Tests the evaluation of the complete pipeline.

    :return:
    """

    eval_args(create_argparser(), ["evaluate"])


@pytest.mark.parametrize("run", range(N))
def test_predict_feeid(run: int):
    """
    Test the frame evoking element predicting of a random raw file.

    NOTE: randomized!

    :param run: Number of random repeats
    :return:
    """

    with RandomFiles(10, True) as m_rndfiles:
        path = m_rndfiles.files[0]
        eval_args(create_argparser(), ["predict", "--feeid", "--path", path])


@pytest.mark.parametrize("run", range(N))
def test_predict_frameid(run: int):
    """
    Test the frame identification of a random raw file.

    NOTE: randomized!

    :param run: Number of random repeats
    :return:
    """

    with RandomFiles(10, True) as m_rndfiles:
        path = m_rndfiles.files[0]
        eval_args(create_argparser(), ["predict", "--frameid", "--path", path])


@pytest.mark.parametrize("run", range(N))
def test_predict_all(run: int):
    """
    Test for a complete pipeline prediction of a random raw file.

    NOTE: randomized!

    :param run: Number of random repeats
    :return:
    """

    with RandomFiles(10, True) as m_rndfiles:
        path = m_rndfiles.files[0]
        eval_args(create_argparser(), ["predict", "--path", path])


def test_config_arg():
    """
    Test the usage of a specified config file.

    :return:
    """

    path = "config.test"
    cM.create_config(path)

    eval_args(create_argparser(), ["evaluate", "--feeid", "--config", path])

    os.remove(path)


@pytest.mark.parametrize("run", range(N))
def test_predict_feeid_out_path(run: int):
    """
    Test the frame evoking element predicting of a random raw file,
    including writing the results to a file.

    NOTE: randomized!

    :param run: Number of random repeats
    :return:
    """

    out_path = create_random_string(string.ascii_lowercase, 8)

    with RandomFiles(10, True) as m_rndfiles:
        path = m_rndfiles.files[0]
        eval_args(
            create_argparser(),
            ["predict", "--feeid", "--path", path, "--out_path", out_path],
        )

    os.remove(out_path)


@pytest.mark.parametrize("run", range(N))
def test_predict_frameid_out_path(run: int):
    """
    Test the frame identification of a random raw file,
    including writing the results to a file.

    NOTE: randomized!

    :param run: Number of random repeats
    :return:
    """

    out_path = create_random_string(string.ascii_lowercase, 8)

    with RandomFiles(10, True) as m_rndfiles:
        path = m_rndfiles.files[0]
        eval_args(
            create_argparser(),
            ["predict", "--frameid", "--path", path, "--out_path", out_path],
        )

    os.remove(out_path)


@pytest.mark.parametrize("run", range(N))
def test_predict_all_out_path(run: int):
    """
    Test for a complete pipeline prediction of a random raw file,
    including writing the results to a file.

    NOTE: randomized!

    :param run: Number of random repeats
    :return:
    """

    out_path = create_random_string(string.ascii_lowercase, 8)

    with RandomFiles(10, True) as m_rndfiles:
        path = m_rndfiles.files[0]
        eval_args(
            create_argparser(), ["predict", "--path", path, "--out_path", out_path]
        )

    os.remove(out_path)

#########################################################
# The following tests are taken from the test_reader file
# as those tests also REQUIRE a whole INSTALLATION they
# were moved here.
#########################################################

def test_semeval_reader_sentences():
    """
    Tests if both systems have read in the exact same sentence data.
    Due to the previous tests for the DataReader this test eases the check for SemevalReader.

    NOTE: Requires semafor and semeval train files!

    :return:
    """

    s_reader = SemevalReader(cM)
    s_reader.read_data(cM.semeval_train[0])

    d_reader = SemaforReader(cM)
    d_reader.read_data(
        cM.train_files[0][0],
        cM.train_files[0][1],
    )

    assert s_reader.sentences == d_reader.sentences


def test_semeval_reader_annotation_size():
    """
    Tests if both systems have read in the same NUMBER of annotations.
    Also the dimensions of the sublists are checked.

    NOTE: Requires semafor and semeval train files!

    :return:
    """

    s_reader = SemevalReader(cM)
    s_reader.read_data(cM.semeval_train[0])

    d_reader = SemaforReader(cM)
    d_reader.read_data(
        cM.train_files[0][0],
        cM.train_files[0][1],
    )

    assert len(s_reader.annotations) == len(d_reader.annotations)

    for s_annos, d_annos in zip(s_reader.annotations, d_reader.annotations):
        assert len(s_annos) == len(d_annos)


def test_semeval_reader_annotations():
    """
    Tests the correctness that both systems have read in the exact same annotations.

    NOTE: Requires semafor and semeval train files!

    :return:
    """

    s_reader = SemevalReader(cM)
    s_reader.read_data(cM.semeval_train[0])

    d_reader = SemaforReader(cM)
    d_reader.read_data(
        cM.train_files[0][0],
        cM.train_files[0][1],
    )

    assert len(s_reader.annotations) == len(d_reader.annotations)

    for s_annos, d_annos in zip(s_reader.annotations, d_reader.annotations):
        for s_anno, d_anno in zip(s_annos, d_annos):
            assert s_anno == d_anno
