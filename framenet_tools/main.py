#!/usr/bin/env python

import argparse
import logging
import os

from typing import List
from subprocess import call

from framenet_tools.config import ConfigManager
from framenet_tools.pipeline import Pipeline
from framenet_tools.utils.static_utils import download, get_spacy_en_model

dirs = ["/scripts", "/lib", "/resources", "/data"]

required_files = [
    "https://github.com/akb89/pyfn/releases/download/v1.0.0/scripts.7z",
    "https://github.com/akb89/pyfn/releases/download/v1.0.0/lib.7z",
    "https://github.com/akb89/pyfn/releases/download/v1.0.0/resources.7z",
    "https://github.com/akb89/pyfn/releases/download/v1.0.0/data.7z",
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
        help=f"Actions to perform, namely: download, convert, train, predict, evaluate",
    )
    parser.add_argument(
        "--feeid", help="Use frame evoking element identification.", action="store_true"
    )
    parser.add_argument(
        "--frameid", help="Use frame identification.", action="store_true"
    )
    parser.add_argument(
        "--spanid", help="Use the span identification.", action="store_true"
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
    parser.add_argument(
        "--batchsize", help="The Batchsize to use for training.", type=int
    )
    parser.add_argument(
        "--config", help="The path to the config file to use.", type=str
    )

    return parser


def eval_args(
    parser: argparse.ArgumentParser, args: List[str] = None
):
    """
    Evaluates the given arguments and runs to program accordingly.

    :param parser: The ArgumentParser for getting the specified arguments
    :param args: Possibility for manually passing arguments.
    :return:
    """

    if args is None:
        parsed = parser.parse_args()
    else:
        parsed = parser.parse_args(args)

    levels = []

    # If a special config file is given, use it
    if parsed.config is not None:
        cM = ConfigManager(parsed.config)
    else:
        # Otherwise try to load the standard file
        cM = ConfigManager('config.file')

    if parsed.feeid:
        levels.append(0)

    if parsed.frameid:
        levels.append(1)

    if parsed.spanid:
        levels.append(2)

    if parsed.batchsize is not None:
        cM.batch_size = parsed.batchsize

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

        pipeline = Pipeline(cM, levels)

        if parsed.use_eval_files:
            pipeline.train(cM.semeval_all)
        else:
            pipeline.train(cM.semeval_train, cM.semeval_dev)

    if parsed.action == "predict":

        if parsed.path is None:
            raise Exception("No input file for prediction given!")

        pipeline = Pipeline(cM, levels)

        pipeline.predict(parsed.path, parsed.out_path)

    if parsed.action == "evaluate":

        pipeline = Pipeline(cM, levels)

        pipeline.evaluate()


def main():
    """
    The main entry point

    :return:
    """

    logging.basicConfig(
        format="%(asctime)s-%(levelname)s-%(message)s", level=logging.INFO
    )

    parser = create_argparser()

    eval_args(parser)
