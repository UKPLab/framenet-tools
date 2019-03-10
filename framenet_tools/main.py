#!/usr/bin/env python

import argparse
import logging
import os

from typing import List
from subprocess import call

# from framenet_tools.frame_identification.feeidentifierWrapper import FeeIdentifierWrapper
from framenet_tools.data_handler.reader import DataReader
from framenet_tools.frame_identification.frameidentifier import FrameIdentifier
from framenet_tools.config import ConfigManager
from framenet_tools.evaluator import (
    evaluate_frame_identification,
    evaluate_fee_identification,
    evaluate_span_identification)
from framenet_tools.frame_identification.utils import download
from framenet_tools.role_identification.spanidentifier import SpanIdentifier

dirs = ["/scripts", "/lib", "/resources", "/data"]

required_files = [
    "https://github.com/akb89/pyfn/releases/download/v1.0.0/scripts.7z",
    "https://github.com/akb89/pyfn/releases/download/v1.0.0/lib.7z",
    "https://github.com/akb89/pyfn/releases/download/v1.0.0/resources.7z",
    "https://github.com/akb89/pyfn/releases/download/v1.0.0/data.7z"
]


def check_files(path):
    logging.info(f"SRLPackage: Checking for required files:")

    for dir, required_file in zip(dirs, required_files):
        complete_path = path + dir

        if os.path.isdir(complete_path):
            logging.info(f"[Skip] Already found {complete_path}!")
        else:
            download(required_file)


def create_argparser():
    """
    Creates the ArgumentParser and defines all of its arguments.

    :return: the set up ArgumentParser
    """

    parser = argparse.ArgumentParser(
        description="Provides tools to self-train and identify frames on raw text."
    )

    parser.add_argument(
        "action",
        help="Actions to perform, namely: download, convert, train, predict, evaluate, fee_predict",
    )
    parser.add_argument(
        "--path", help="A path specification used by some actions.", type=str
    )
    parser.add_argument(
        "--out_path", help="The path used for saving predictions", type=str
    )
    parser.add_argument(
        "--use_eval_files",
        help="Specify if eval files should be used for training as well.",
        action="store_true",
    )

    return parser


def eval_args(
    parser: argparse.ArgumentParser, cM: ConfigManager, args: List[str] = None
):
    """
    Evaluates the given arguments and runs to program accordingly.

    :param parser: The ArgumentParser for getting the specified arguments
    :param cM: The ConfigManager for getting necessary variables
    :return:
    """

    if args is None:
        parsed = parser.parse_args()
    else:
        parsed = parser.parse_args(args)

    if parsed.action == "download":

        if parsed.path is not None:
            check_files(os.path.join(os.getcwd(), parsed.download))
        else:
            check_files(os.getcwd())

    if parsed.action == "convert":

        call(
            [
                "pyfn",
                "convert",
                "--from",
                "fnxml",
                "--to",
                "semafor",
                "--source",
                "data/fndata-1.5-with-dev",
                "--target",
                "data/experiments/xp_001/data",
                "--splits",
                "train",
                "--output_sentences",
            ]
        )

    if parsed.action == "train":

        f_i = FrameIdentifier(cM)

        if parsed.use_eval_files:
            f_i.train(cM.all_files)
        else:
            f_i.train(cM.train_files)

        f_i.save_model(cM.saved_model)

    if parsed.action == "predict":

        if parsed.path is None:
            raise Exception("No input file for prediction given!")

        if parsed.out_path is None:
            raise Exception("No path specified for saving!")

        f_i = FrameIdentifier(cM)
        f_i.load_model(cM.saved_model)
        f_i.write_predictions(parsed.path, parsed.out_path)

    if parsed.action == "fee_predict":

        if parsed.path is None:
            raise Exception("No input file for prediction given!")

        if parsed.out_path is None:
            raise Exception("No path specified for saving!")

        f_i = FrameIdentifier(cM)
        f_i.write_predictions(parsed.path, parsed.out_path, fee_only=True)

    if parsed.action == "evaluate":

        evaluate_frame_identification(cM)

    if parsed.action == "fee_evaluate":

        evaluate_fee_identification(cM)


def main():
    """
    The main entry point

    :return:
    """

    logging.basicConfig(
        format="%(asctime)s-%(levelname)s-%(message)s", level=logging.DEBUG
    )

    cM = ConfigManager()
    parser = create_argparser()

    eval_args(parser, cM)


cM = ConfigManager()
parser = create_argparser()

logging.basicConfig(format="%(asctime)s-%(levelname)s-%(message)s", level=logging.DEBUG)

# eval_args(parser, cM, ["train"])
# evaluate_span_identification(cM)

file = cM.train_files[0]
m_data_reader = DataReader(cM)
m_data_reader.read_data(file[0], file[1])

span_identifier = SpanIdentifier(cM)
span_identifier.train(m_data_reader.sentences, m_data_reader.annotations)
evaluate_span_identification(cM, span_identifier)
# eval_args(parser, cM, ["convert"])

#

# eval_args(parser, cM, ["predict", "--path", "data/example.txt", "--out_path", "data/test.json"])

# cM = ConfigManager()
# f_i = FrameIdentifier(cM)

# f_i.load_model(cM.saved_model)
# f_i.write_predictions("data/example.txt", "data/out_fee.txt", True)

# f_i.train(cM.train_files)
# print(f_i.evaluate_file(cM.eval_files[0]))
# f_i.save_model(cM.saved_model)
# evaluate_frame_identification(cM)
# print(evaluate_fee_identification(cM.eval_files[0]))

# fiw = FeeIdentifierWrapper(cM)
# fiw.train(cM.train_files)
# print(fiw)
# print(evaluate_fee_identification(cM.eval_files[0], fiw))
