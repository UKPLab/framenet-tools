import pytest

from framenet_tools.tests.test_reader import RandomFiles


# Parameter to run tests N times.
N = 10


@pytest.mark.parametrize('run', range(N))
def test_load_fees(run: int):
    """

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

    :return:
    """

    with RandomFiles(10, True) as m_rndfiles:
        annotations = m_rndfiles.m_reader.annotations
        annotations_length = sum([len(x) for x in annotations])

        assert annotations_length == 0


@pytest.mark.parametrize('run', range(N))
def test_predict_fees(run: int):
    """

    :return:
    """

    with RandomFiles(10, True) as m_rndfiles:

        m_rndfiles.m_reader.predict_fees()

        annotations = m_rndfiles.m_reader.annotations
        annotations_length = sum([len(x) for x in annotations])

        assert annotations_length > 0


@pytest.mark.parametrize('run', range(N))
def test_predict_loaded_fees(run: int):
    """

    :return:
    """

    with RandomFiles(100, False) as m_rndfiles:

        annotations = m_rndfiles.m_reader.annotations
        old_annotations_length = sum([len(x) for x in annotations])

        m_rndfiles.m_reader.predict_fees()
        annotations = m_rndfiles.m_reader.annotations
        annotations_length = sum([len(x) for x in annotations])

        assert annotations_length != old_annotations_length



