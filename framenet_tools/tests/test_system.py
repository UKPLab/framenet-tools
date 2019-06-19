import os
import pytest
from framenet_tools.config import ConfigManager
from framenet_tools.main import eval_args, create_argparser
from framenet_tools.tests.test_reader import RandomFiles

cM = ConfigManager("config.file")

cM.semeval_train = ["../../" + cM.semeval_train[0]]
cM.semeval_dev = ["../../" + cM.semeval_dev[0]]
cM.semeval_test = ["../../" + cM.semeval_test[0]]

cM.num_epochs = 1

# Parameter to run tests N times.
N = 4


"""
WARNING: The following tests check complete functionality of the system!

Therefore running a variety of complete training and evaluation tasks.
This can take a long time!

Also note that each test evaluates exactly one argset instead of programmatically 
do it all at once, for better overview and debugging.
"""


def test_train_feeid():
    """
    Test for training the frame evoking element identification

    NOTE: This stage can not be trained, though for completeness it needs to be tested.

    :return:
    """

    eval_args(create_argparser(), cM, ["train", "--feeid"])


def test_train_frameid():
    """
    Test for training the frame identification

    NOTE: Might take a long time!

    :return:
    """

    eval_args(create_argparser(), cM, ["train", "--frameid"])


def test_train():
    """
    Test for training the complete pipeline

    NOTE: Might take a long time!

    :return:
    """

    eval_args(create_argparser(), cM, ["train"])


def test_eval_feeid():
    """
    Test for evaluating the frame evoking element identification.

    :return:
    """

    eval_args(create_argparser(), cM, ["evaluate", "--feeid"])


def test_eval_frameid():
    """
    Test the evaluation of the frame identification.

    :return:
    """

    eval_args(create_argparser(), cM, ["evaluate", "--frameid"])


def test_eval_all():
    """
    Tests the evaluation of the complete pipeline.

    :return:
    """

    eval_args(create_argparser(), cM, ["evaluate"])


@pytest.mark.parametrize('run', range(N))
def test_predict_feeid(run: int):
    """
    Test the frame evoking element predicting of a random raw file.

    NOTE: randomized!

    :param run: Number of random repeats
    :return:
    """

    with RandomFiles(10, True) as m_rndfiles:
        path = m_rndfiles.files[0]
        eval_args(create_argparser(), cM, ["predict", "--feeid", "--path", path])


@pytest.mark.parametrize('run', range(N))
def test_predict_frameid(run: int):
    """
    Test the frame identification of a random raw file.

    NOTE: randomized!

    :param run: Number of random repeats
    :return:
    """

    with RandomFiles(10, True) as m_rndfiles:
        path = m_rndfiles.files[0]
        eval_args(create_argparser(), cM, ["predict", "--frameid", "--path", path])


@pytest.mark.parametrize('run', range(N))
def test_predict_all(run: int):
    """
    Test for a complete pipeline prediction of a random raw file.

    NOTE: randomized!

    :param run: Number of random repeats
    :return:
    """

    with RandomFiles(10, True) as m_rndfiles:
        path = m_rndfiles.files[0]
        eval_args(create_argparser(), cM, ["predict", "--path", path])
