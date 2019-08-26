Usage
=====

| The following functions both require a pretrained model,
| generate using ``framenet_tools train`` as explained above. - Stages:
  The System is split into 4 distinct pipeline stages, namely: - 1 Frame
  evoking element identification - 2 Frame identification - 3 Span
  identification (WIP) - 4 Role identification (WIP)

Each stage can individually be trained by calling it
e.g. \ ``--frameid``. Also combinations of mutliple stages are possible.
This can be done for every option. NOTE: A usage of ``evaluate`` or
``predict`` requires a previous training of the same stage level!

-  ``framenet_tools predict --path [path]``
   annotates the given raw text file located at ``--path`` and prints
   the result. Optionally ``--out_path`` can be used to write the
   results directly to a file. Also a prediction can be limited to a
   certain stage by specifying it (e.g. ``--feeid``). NOTE: As the
   stages build on the previous ones, this option represents a upper
   bound.
-  ``framenet_tools evaluate``
   evaluates the F1-Score of the model on the evaluation files. Here,
   evaluation can be exclusively limited to a certain stage.

Logging
-------

Training automatically logs the loss and accuracy of the train- and
devset in `TensorBoard`_ format. - ``tensorboard --logdir=runs`` can be
used to run TensorBoard and visualize the data.

Formats
-------

Currently support formats include:

-  Raw text
-  SEMEVAL XML: the format of the SEMEVAL 2007 shared task 19 on frame
   semantic structure extraction
-  SEMAFOR CoNLL: the format used by the SEMAFOR parser

NOTE: If the format is not supported, `pyfn`_ might be providing a
conversion.

.. _TensorBoard: https://www.tensorflow.org/guide/summaries_and_tensorboard
.. _pyfn: https://pypi.org/project/pyfn/
