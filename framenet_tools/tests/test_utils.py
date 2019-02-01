import pytest

from framenet_tools.evaluator import calc_f


@pytest.mark.parametrize(
    "testdata, expected",
    [([0] * 3, [0] * 3), ([1, 0, 0], [1, 1, 1]), ([1, 1, 1], [0.5, 0.5, 0.5])],
)
def test_calc_f(testdata: list, expected: list):
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
