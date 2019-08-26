Installation
============

- Clone repository or download files
- Enter the directory
- Run: ``pip install -e .``

Setup
-----

-  ``framenet_tools download``
   acquires all required data and extracts it , optionally ``--path``
   can be used to specify a custom path; default is the current
   directory.
   NOTE: After extraction the space occupied amounts up to around 9GB!
-  ``framenet_tools convert``
   can now be used to generate the CoNLL datasets
   This function is analogous to pyfn and simply propagates the call.
-  ``framenet_tools train``
   trains a new model on the training files and saves it,
   optionally ``--use_eval_files`` can be specified to train on the
   evaluation files as well.
   NOTE: Training can take a few minutes, depending on the hardware.

For further information run ``framenet_tools --help``

Alternative
^^^^^^^^^^^

Alternatively conversion.sh provides a also the ability to convert FN
data to CoNLL using pyfn. In this case, manually download and extract
the `FrameNet dataset`_ and adjust the path inside the script.

.. _FrameNet dataset: https://github.com/akb89/pyfn/releases/download/v1.0.0/data.7z