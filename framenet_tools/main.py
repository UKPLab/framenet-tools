#!/usr/bin/env python

import argparse
import logging
import os
from subprocess import call
from typing import List

#from framenet_tools.frame_identification.feeidentifierWrapper import FeeIdentifierWrapper
from framenet_tools.frame_identification.frameidentifier import FrameIdentifier
from framenet_tools.config import ConfigManager
from framenet_tools.evaluator import (
    evaluate_frame_identification,
    evaluate_fee_identification,
)
from framenet_tools.frame_identification.utils import extract7z, download_file


def extract_file(file_path: str):
    """
    Extracts a zipped file
    :param file_path: The file to extract
    :return:
    """
    call(["7z", "x", file_path])

    # TODO extract using python, NOTE ran into trouble because of 7z
    # raw = open(file_path,"rb")
    # archive = Archive7z(raw)
    # data = archive.getmember(archive.getnames()[0]).read()
    # raw.close()

    # Cleanup
    os.remove(file_path)


def download_scripts():
    """
    Helper function for downloading the scripts

    NOTE : The paths should NOT be changed in order for pyfn to work

    :return:
    """
    url = "https://github.com/akb89/pyfn/releases/download/v1.0.0/scripts.7z"
    file_path = "scripts.7z"
    print("Downloading scripts:")

    download_file(url, file_path)
    extract_file(file_path)


def download_lib():
    url = "https://github.com/akb89/pyfn/releases/download/v1.0.0/lib.7z"
    file_path = "lib.7z"
    print("Downloading lib:")

    download_file(url, file_path)
    extract_file(file_path)


def download_resources():
    url = "https://github.com/akb89/pyfn/releases/download/v1.0.0/resources.7z"
    file_path = "resources.7z"
    print("Downloading resources:")

    download_file(url, file_path)
    extract_file(file_path)


def download_data():
    url = "https://github.com/akb89/pyfn/releases/download/v1.0.0/data.7z"
    file_path = "data.7z"
    print("Downloading data:")

    download_file(url, file_path)
    extract_file(file_path)


def check_files(path):
    print("SRLPackage: Checking for required files:")
    script_path = path + "/scripts"
    lib_path = path + "/lib"
    resources_path = path + "/resources"
    data_path = path + "/data"

    if os.path.isdir(script_path):
        print("[Skip] Already found scripts!")
    else:
        # print(script_path)
        download_scripts()

    if os.path.isdir(lib_path):
        print("[Skip] Already found lib!")
    else:
        # print("d")
        download_lib()

    if os.path.isdir(resources_path):
        print("[Skip] Already found resources!")
    else:
        # print("d")
        download_resources()

    if os.path.isdir(data_path):
        print("[Skip] Already found data!")
    else:
        # print("d")
        download_data()


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


def eval_args(parser: argparse.ArgumentParser, cM: ConfigManager, args: List[str] = None):
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


def main():
    """
    The main entry point

    :return:
    """

    logging.basicConfig(format='%(asctime)s-%(levelname)s-%(message)s', level=logging.DEBUG)

    cM = ConfigManager()
    parser = create_argparser()

    eval_args(parser, cM)


# cM = ConfigManager()
# parser = create_argparser()

# logging.basicConfig(format='%(asctime)s-%(levelname)s-%(message)s', level=logging.DEBUG)

# eval_args(parser, cM, ["train"])

# eval_args(parser, cM, ["download"])

#eval_args(parser, cM, ["predict", "--path", "data/example.txt", "--out_path", "data/test.json"])

#cM = ConfigManager()
#f_i = FrameIdentifier(cM)

#f_i.load_model(cM.saved_model)
#f_i.write_predictions("data/example.txt", "data/out_fee.txt", True)

#f_i.train(cM.train_files)
#print(f_i.evaluate_file(cM.eval_files[0]))
#f_i.save_model(cM.saved_model)
#evaluate_frame_identification(cM)
#print(evaluate_fee_identification(cM.eval_files[0]))

#fiw = FeeIdentifierWrapper(cM)
#fiw.train(cM.train_files)
#print(fiw)
#print(evaluate_fee_identification(cM.eval_files[0], fiw))