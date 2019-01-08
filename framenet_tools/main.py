#!/usr/bin/env python
import os
import requests
import logging
import sys
import pyfn
from subprocess import call
from framenet_tools.frame_identification.frameidentifier import FrameIdentifier
from framenet_tools.paths import *
from framenet_tools.evaluator import evaluate_frame_identification, evaluate_fee_identification


def download_file(url: str, file_path: str):
    """
    Downloads a file and saves at a given path

    :param url: The URL of the file to download
    :param file_path: The destination of the file
    :return:
    """

    r = requests.get(url, stream=True)
    with open(file_path, "wb") as fd:
        logging.info("Downloading [%s] and saving to [%s]", r.url, file_path)
        for chunk in r.iter_content(chunk_size=10 * 1024 * 1024):
            fd.write(chunk)


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
        download_scripts()

    if os.path.isdir(lib_path):
        print("[Skip] Already found lib!")
    else:
        download_lib()

    if os.path.isdir(resources_path):
        print("[Skip] Already found resources!")
    else:
        download_resources()

    if os.path.isdir(data_path):
        print("[Skip] Already found data!")
    else:
        download_data()


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] in ["help", "-h", "--help"]:
            print(
                "SRLPackage usage:\n download - downloads and extracts all required files \
				\n convert - converts the data to CoNLL format (analogous to pyfn's convert)"
            )
        if sys.argv[1] in ["download"]:
            path = os.getcwd()
            check_files(path)
        if sys.argv[1] in ["convert"]:
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
        if sys.argv[1] in ["train"]:
            f_i = FrameIdentifier()
            f_i.train_complete()
            f_i.save_model("model_name")

        if sys.argv[1] in ["predict"]:
            f_i = FrameIdentifier()
            f_i.load_model("model_name")
            f_i.write_predictions(sys.argv[2], sys.argv[3])

        if sys.argv[1] in ["evaluate"]:
            f1 = evaluate_frame_identification("model_name", DEV_FILES)
            print(f1)


# main()
