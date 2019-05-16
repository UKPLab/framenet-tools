#!/usr/bin/env python

import argparse
import logging
import os

from typing import List
from subprocess import call

from framenet_tools.frame_identification.frameidentifier import FrameIdentifier
from framenet_tools.config import ConfigManager
from framenet_tools.evaluator import (
    evaluate_frame_identification,
    evaluate_fee_identification,
    evaluate_span_identification)
from framenet_tools.utils.static_utils import download, get_spacy_en_model
from framenet_tools.span_identification.spanidentifier import SpanIdentifier

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
        help=f"Actions to perform, namely: download, convert, train, predict, evaluate, fee_predict, span_evaluate",
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

        get_spacy_en_model()

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

        for dataset in ["train", "dev", "test"]:
            call(
                [
                    "pyfn",
                    "convert",
                    "--from",
                    "fnxml",
                    "--to",
                    "semeval",
                    "--source",
                    "data/fndata-1.5-with-dev",
                    "--target",
                    "data/experiments/xp_001/data",
                    "--splits",
                    dataset,
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

    if parsed.action == "span_evaluate":

        evaluate_span_identification(cM)


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


logging.basicConfig(
        format="%(asctime)s-%(levelname)s-%(message)s", level=logging.INFO
    )

cM = ConfigManager()
cM.fEM.read_frame_embeddings()

print(cM.fEM.frames["Practice"])
cM.wEM.read_word_embeddings()

parser = create_argparser()

file = "data/experiments/xp_001/data/train.gold.xml"
#m_data_reader = SemevalReader(cM)
#m_data_reader.read_data(file)

#m_data_reader.generate_pos_tags()

#m_data_reader.embed_words()
#m_data_reader.embed_frames()

#file = cM.semeval_files[0]
#m_data_reader_dev = SemevalReader(cM)
#m_data_reader_dev.read_data(file)

#m_data_reader_dev.generate_pos_tags()
#m_data_reader_dev.embed_words()
#m_data_reader_dev.embed_frames()

# exit()
#create_lexicon()

#lex = load_lexicon("data/lexicon.file")

#print(lex)


span_identifier = SpanIdentifier(cM)
span_identifier.load()
#span_identifier.train(m_data_reader, m_data_reader_dev)
evaluate_span_identification(cM, span_identifier)

