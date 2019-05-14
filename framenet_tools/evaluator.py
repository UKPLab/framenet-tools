import logging

from copy import deepcopy

from framenet_tools.config import ConfigManager
from framenet_tools.data_handler.semeval_reader import SemevalReader
from framenet_tools.frame_identification.frameidentifier import FrameIdentifier
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


def evaluate_span_identification(cM: ConfigManager, span_identifier: SpanIdentifier = None):
    """
    Evaluates the span identification for its F1 score

    :param cM: The ConfigManager containing the evaluation files
    :param span_identifier: Optionally an instance of a SpanIdentifier
    :return: A Triple of Precision, Recall and F1-Score
    """

    logging.info(f"Evaluating Span Identification:")

    for file in cM.semeval_files:

        logging.info(f"Evaluating on: {file}")

        m_data_reader = SemevalReader(cM)
        m_data_reader.read_data(file)
        m_data_reader.embed_words()
        m_data_reader.embed_frames()

        gold_sentences = deepcopy(m_data_reader.annotations)

        m_data_reader.predict_spans(span_identifier)

        tp = fp = fn = 0

        for i in range(len(m_data_reader.sentences)):

            gold_annotations = gold_sentences[i]
            predictied_annotations = m_data_reader.annotations[i]

            for gold_annotation, predictied_annotation in zip(gold_annotations, predictied_annotations):

                for role_posistion in gold_annotation.role_positions:
                    if role_posistion in predictied_annotation.role_positions:
                        tp += 1
                    else:
                        fn += 1

                for role_posistion in predictied_annotation.role_positions:
                    if role_posistion not in gold_annotation.role_positions:
                        fp += 1

        pr, re, f1 = calc_f(tp, fp, fn)

        logging.info(f"FEE Evaluation complete!\n"
                     f"True Positives: {tp} False Postives: {fp} False Negatives: {fn} \n"
                     f"Precision: {pr} Recall: {re} F1-Score: {f1}")

    return pr, re, f1


def evaluate_fee_identification(cM: ConfigManager):
    """
    Evaluates the F1-Score of the Frame Evoking Element Identification only

    :param cM: The ConfigManager containing the saved model and evaluation files
    :return: A Triple of Precision, Recall and F1-Score
    """

    for file in cM.eval_files:

        logging.info(f"Evaluating on: {file[0]}")

        m_data_reader = DataReader(cM)
        m_data_reader.read_data(file[0], file[1])

        gold_sentences = m_data_reader.annotations.copy()

        m_data_reader.predict_fees()

        tp = fp = fn = 0

        for gold_annotations, predictied_annotations in zip(
            gold_sentences, m_data_reader.annotations
        ):
            for gold_annotation in gold_annotations:
                if gold_annotation.fee_raw in [x.fee_raw for x in predictied_annotations]:
                    tp += 1
                else:
                    fn += 1

            for predicted_annotation in predictied_annotations:
                #print(predicted_annotation)
                #print([x.fee_raw for x in gold_annotations])
                if predicted_annotation.fee_raw not in [x.fee_raw for x in gold_annotations]:
                    fp += 1

        pr, re, f1 = calc_f(tp, fp, fn)

        logging.info(f"FEE Evaluation complete!\n"
                     f"True Positives: {tp} False Postives: {fp} False Negatives: {fn} \n"
                     f"Precision: {pr} Recall: {re} F1-Score: {f1}")

    return pr, re, f1


def evaluate_frame_identification(cM: ConfigManager):
    """
    Evaluates the F1-Score for a model on a given file set

    :param cM: The ConfigManager containing the saved model and evaluation files
    :return: A Triple of Precision, Recall and F1-Score
    """

    f_i = FrameIdentifier(cM)
    f_i.load_model(cM.saved_model)

    for file in cM.eval_files:
        logging.info(f"Evaluating on: {file[0]}")
        tp, fp, fn = f_i.evaluate_file(file)
        pr, re, f1 = calc_f(tp, fp, fn)

        logging.info(f"Evaluation complete!\n"
                     f"True Positives: {tp} False Postives: {fp} False Negatives: {fn} \n"
                     f"Precision: {pr} Recall: {re} F1-Score: {f1}")

    return pr, re, f1


"""
f1 = evaluate_fee_identification(DEV_FILES)
print(f1)

f1 = evaluate_frame_identification(SAVED_MODEL, DEV_FILES)
print(f1)

f_i = FrameIdentifier()
f_i.load_model(SAVED_MODEL)
f_i.write_predictions("../data/experiments/xp_001/data/WallStreetJournal20150915.txt", "here.txt")
"""
