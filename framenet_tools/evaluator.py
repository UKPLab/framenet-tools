import logging

from copy import deepcopy
from typing import List

from framenet_tools.config import ConfigManager
from framenet_tools.data_handler.semevalreader import SemevalReader
from framenet_tools.frame_identification.frameidentifier import get_dataset
from framenet_tools.data_handler.reader import DataReader
from framenet_tools.span_identification.spanidentifier import SpanIdentifier


def calc_f(tp: int, fp: int, fn: int):
    """
    Calculates the F1-Score

    NOTE: This follows standard evaluation metrics
    TAKEN FROM: Open-SESAME (https://github.com/swabhs/open-sesame)

    :param tp: True Postivies Count
    :param fp: False Postivies Count
    :param fn: False Negatives Count
    :return: A Triple of Precision, Recall and F1-Score
    """
    if tp == 0.0 and fp == 0.0:
        pr = 0.0
    else:
        pr = tp / (tp + fp)
    if tp == 0.0 and fn == 0.0:
        re = 0.0
    else:
        re = tp / (tp + fn)
    if pr == 0.0 and re == 0.0:
        f = 0.0
    else:
        f = 2.0 * pr * re / (pr + re)
    return pr, re, f


def evaluate_span_identification(m_reader: DataReader, original_reader: DataReader):
    """
    Evaluates the span identification for its F1 score

    :param m_reader: The reader containing the predicted annotations
    :param original_reader: The original reader containing the gold annotations
    :return: A Triple of True positives, False positives and False negatives
    """

    tp = fp = fn = 0

    for i in range(len(m_reader.sentences)):

        gold_annotations = original_reader.annotations[i]
        predictied_annotations = m_reader.annotations[i]

        for gold_annotation, predictied_annotation in zip(
            gold_annotations, predictied_annotations
        ):

            for role_posistion in gold_annotation.role_positions:
                if role_posistion in predictied_annotation.role_positions:
                    tp += 1
                else:
                    fn += 1

            for role_posistion in predictied_annotation.role_positions:
                if role_posistion not in gold_annotation.role_positions:
                    fp += 1

    return tp, fp, fn


def evaluate_fee_identification(m_reader: DataReader, original_reader: DataReader):
    """
    Evaluates the Frame Evoking Element Identification only

    :param m_reader: The reader containing the predicted annotations
    :param original_reader: The original reader containing the gold annotations
    :return: A Triple of True positives, False positives and False negatives
    """

    gold_sentences = original_reader.annotations.copy()

    tp = fp = fn = 0

    for gold_annotations, predictied_annotations in zip(
        gold_sentences, m_reader.annotations
    ):
        for gold_annotation in gold_annotations:
            if gold_annotation.fee_raw in [x.fee_raw for x in predictied_annotations]:
                tp += 1
            else:
                fn += 1

        for predicted_annotation in predictied_annotations:
            if predicted_annotation.fee_raw not in [
                x.fee_raw for x in gold_annotations
            ]:
                fp += 1

    return tp, fp, fn


def evaluate_frame_identification(m_reader: DataReader, original_reader: DataReader):
    """
    Evaluates the Frame Identification

    :param m_reader: The reader containing the predicted annotations
    :param original_reader: The original reader containing the gold annotations
    :return: A Triple of True positives, False positives and False negatives
    """

    # Load correct answers for comparison:
    gold_xs, gold_ys = get_dataset(original_reader)
    xs, ys = get_dataset(m_reader)
    tp = 0
    fp = 0
    fn = 0

    found = False

    for gold_x, gold_y in zip(gold_xs, gold_ys):
        for x, y in zip(xs, ys):
            if gold_x == x and gold_y == y:
                found = True
                break

        if found:
            tp += 1
        else:
            fn += 1

        found = False

    for x, y in zip(xs, ys):
        for gold_x, gold_y in zip(gold_xs, gold_ys):
            if gold_x == x and gold_y == y:
                found = True

        if not found:
            fp += 1

        found = False

    return tp, fp, fn


def evaluate_stages(
    m_reader: DataReader, original_reader: DataReader, levels: List[int]
):
    """
    Evaluates the stages specified in levels

    :param m_reader: The reader including the predicted data
    :param original_reader: The reader which holds the gold data
    :param levels: The levels to evaluate for
    :return: A triple of Precision, Recall and the F1-Score
    """

    if max(levels) == 0:
        tp, fp, fn = evaluate_fee_identification(m_reader, original_reader)

    if max(levels) == 1:
        tp, fp, fn = evaluate_frame_identification(m_reader, original_reader)

    if max(levels) == 2:
        tp, fp, fn = evaluate_span_identification(m_reader, original_reader)

    pr, re, f1 = calc_f(tp, fp, fn)

    logging.info(
        f"Evaluation complete!\n"
        f"True Positives: {tp} False Postives: {fp} False Negatives: {fn} \n"
        f"Precision: {pr} Recall: {re} F1-Score: {f1}"
    )

    return pr, re, f1
