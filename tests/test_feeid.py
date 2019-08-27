import pytest

from framenet_tools.config import ConfigManager
from framenet_tools.stages.feeID import FeeID
from tests.test_reader import RandomFiles


# Parameter to run tests N times.
N = 10


@pytest.mark.parametrize('run', range(N))
def test_load_fees(run: int):
    """
    Tests whether all fees are loaded from a file.

    NOTE: randomized

    :param run: Number of random repeats
    :return:
    """

    with RandomFiles(10, False) as m_rndfiles:

        annotations = m_rndfiles.m_reader.annotations

        with open(m_rndfiles.files[1]) as file:
            raw = file.read()

        raw = raw.rsplit("\n")

        raw.remove("")

        annotations_length = sum([len(x) for x in annotations])

        assert annotations_length == len(raw)


@pytest.mark.parametrize('run', range(N))
def test_no_fees(run: int):
    """
    Tests whether a raw file is loaded without any fees.

    NOTE: randomized

    :param run: Number of random repeats
    :return:
    """

    with RandomFiles(10, True) as m_rndfiles:
        annotations = m_rndfiles.m_reader.annotations
        annotations_length = sum([len(x) for x in annotations])

        assert annotations_length == 0


@pytest.mark.parametrize('run', range(N))
def test_predict_fees(run: int):
    """
    Tests if for any random raw file there is some fee prediction.

    NOTE: randomized

    :param run: Number of random repeats
    :return:
    """

    with RandomFiles(10, True) as m_rndfiles:

        m_feeID = FeeID(ConfigManager('config.file'))

        m_feeID.predict(m_rndfiles.m_reader)

        annotations = m_rndfiles.m_reader.annotations
        annotations_length = sum([len(x) for x in annotations])

        assert annotations_length > 0


@pytest.mark.parametrize('run', range(N))
def test_predict_loaded_fees(run: int):
    """
    Tests if for any random annotated file, there are some fees predicted.

    NOTE: randomized

    :param run: Number of random repeats
    :return:
    """

    with RandomFiles(100, False) as m_rndfiles:

        annotations = m_rndfiles.m_reader.annotations
        old_annotations_length = sum([len(x) for x in annotations])

        m_feeID = FeeID(ConfigManager('config.file'))

        m_feeID.predict(m_rndfiles.m_reader)
        annotations = m_rndfiles.m_reader.annotations
        annotations_length = sum([len(x) for x in annotations])

        assert annotations_length != old_annotations_length
        assert annotations_length > 0



