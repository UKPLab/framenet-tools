# framenet-tools

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![Documentation Status](https://readthedocs.org/projects/framenet-tools/badge/?version=latest)](https://framenet-tools.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/inception-project/framenet-tools.svg?branch=master)](https://travis-ci.org/inception-project/framenet-tools)
[![codecov](https://codecov.io/gh/inception-project/framenet-tools/branch/master/graph/badge.svg)](https://codecov.io/gh/inception-project/framenet-tools)


Provides functionality to find Frame Evoking Elements in raw text and predict 
their corresponding frames. Furthermore possible spans of roles can be found and assigned. 
Models can be trained either on the given files or on any annotated file in a supported format (For more information
look at the section formats).  
- Based on (and using) [pyfn][1]

## Installation
- Clone repository or download files
- Enter the directory
- Run `pip install -e .`

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
- Stages: The System is split into 4 distinct pipeline stages, namely:
    - 1 Frame evoking element identification
    - 2 Frame identification
    - 3 Span identification (WIP)
    - 4 Role identification (WIP)

Each stage can individually be trained by calling it e.g. `--frameid`.
Also combinations of mutliple stages are possible. This can be done for every option.
NOTE: A usage of `evaluate` or `predict` requires a previous training of the same stage level! 
    
- `framenet_tools predict --path [path]`  
annotates the given raw text file located at
 `--path` and prints the result. Optionally `--out_path` can be used to write the results directly to a file.
 Also a prediction can be limited to a certain stage by specifying it (e.g. `--feeid`). NOTE: As the stages build 
on the previous ones, this option represents a upper bound. 
- `framenet_tools evaluate`  
evaluates the F1-Score of the model on the evaluation files.
Here, evaluation can be exclusively limited to a certain stage.

## Logging

Training automatically logs the loss and accuracy of the train- and devset in [TensorBoard][3] format. 
- `tensorboard --logdir=runs`
can be used to run TensorBoard and visualize the data.

## Architecture

![alt text](Overview.png "Architecture")

[1]: https://pypi.org/project/pyfn/
[2]: https://github.com/akb89/pyfn/releases/download/v1.0.0/data.7z
[3]: https://www.tensorflow.org/guide/summaries_and_tensorboard

## Formats

Currently support formats include:

- Raw text
- SEMEVAL XML: the format of the SEMEVAL 2007 shared task 19 on frame semantic structure extraction
- SEMAFOR CoNLL: the format used by the SEMAFOR parser

NOTE: If the format is not supported, [pyfn][1] might be providing a conversion.

## Citing & Authors
If you find this repository helpful, feel free to cite

```
@software{andre_markard_2020_3993970,
  author       = {André Markard and
                  Jan-Christoph Klie},
  title        = {{FrameNet Tools: A Python library to work with 
                   FrameNet}},
  month        = aug,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {v0.1.0},
  doi          = {10.5281/zenodo.3993970},
  url          = {https://doi.org/10.5281/zenodo.3993970}
}
```
