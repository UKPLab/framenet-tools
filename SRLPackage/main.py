#!/usr/bin/env python
import os
import requests
import logging
from subprocess import call

def download_file(url, file_path):
	r = requests.get(url, stream=True)
	with open(file_path, 'wb') as fd:
		logging.info('Downloading [%s] and saving to [%s]', r.url, file_path)
		for chunk in r.iter_content(chunk_size=10 * 1024 * 1024):
			fd.write(chunk)

def extract_file(file_path):
	call(["7z", "x", file_path])

	#TODO extract using python, NOTE ran into trouble
	#raw = open(file_path,"rb")
	#archive = Archive7z(raw)
	#data = archive.getmember(archive.getnames()[0]).read()
	#raw.close()

	#Cleanup
	os.remove(file_path) 

def download_scripts():
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
	print("Downloading resources:")
	
	download_file(url, file_path)
	extract_file(file_path)


def check_files():
	print("SRLPackage: Checking for required files:")
	script_path = os.path.dirname(os.path.realpath(__file__)) + "/scripts"
	lib_path = os.path.dirname(os.path.realpath(__file__)) + "/lib"
	resources_path = os.path.dirname(os.path.realpath(__file__)) + "/resources"
	resources_path = os.path.dirname(os.path.realpath(__file__)) + "/data"

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
	check_files()


main()
