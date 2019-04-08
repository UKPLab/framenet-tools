import setuptools

#######################
# Taken from:
# https://packaging.python.org/tutorials/packaging-projects/
#######################

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="framenet_tools",
    version="0.0.1",
    # scripts=["SRLPackage"],
    author="Andre Markard",
    author_email="andre@markard.eu",
    description="A summarization of the SRL process",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="Yet to be published",
    platforms=["any"],
    packages=["framenet_tools",
              "framenet_tools.frame_identification",
              ],
    install_requires=["pyfn",
                      "torch",
                      "torchtext",
                      "nltk",
                      "requests",
                      "scipy",
                      "spacy",
                      "tensorboardX",
                      "tqdm",
                      "numpy",
                      "jsondiff",
                      "jsonnet",
                      "jsonpickle",
                      "h5py",
                      "pylzma",
                      "pytest",
                      # "allennlp",
                      # "flair",
                      ],
    entry_points={"console_scripts": ["framenet_tools = framenet_tools.main:main"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Apache License, Version 2.0",
        "Operating System :: OS Independent",
    ],
)
