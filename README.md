# framenet-tools

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)


A summarization of the SRL process.  
Provides functionality to find Frame Evoking Elements in raw text and predict 
their corresponding frames.  
Models can be trained either on given files or on any annotated file in CoNLL format.  
- Based on (and using) [pyfn][1]

## Installation
- Clone repository or download files
- Enter the directory
- Run `pip3 install .`

## Setup
- `framenet_tools download`  
acquires all required data and extracts it
, optionally `--path` can be used 
to specify a custom path; default is the current directory.  
NOTE: After extraction the space occupied amounts up to around 9GB!
- `framenet_tools convert`  
can now be used to generate the CoNLL datasets  
This function is analogous to pyfn and simply propagates the call.
- `framenet_tools train`  
trains a new model on the training files and saves it,  
optionally `--use_eval_files` can be specified to train on the evaluation files as well.  
NOTE: Training can take a few minutes, depending on the hardware. 

For further information run `framenet_tools --help`

#### Alternative
Alternatively conversion.sh provides a also the ability to convert FN data to CoNLL using pyfn.
In this case, manually download and extract the [FrameNet dataset][2]
and adjust the path inside the script.

## Usage

The following functions both require a pretrained model,  
generate using `framenet_tools train` as explained above.
- `framenet_tools predict --path --out_path`  
annotates the given raw text file located at
 `--path` and writes the output to `--out_path`
- `framenet_tools evaluate`  
evaluates the F1-Score of the model on the evaluation files.



[1]: https://pypi.org/project/pyfn/
[2]: https://github.com/akb89/pyfn/releases/download/v1.0.0/data.7z