import lzma
import pickle

from framenet_tools.lexicon.lexicon import LexiconDataSet


"""
The Lexicon Creator

Run this script to create the lexicons needed by different models.
NOTE: Not only the lexicon baselines require the creation of the 
lexcions in order to work. 
"""


lexiconFile = "lexicon.file"


def create_lexicon(lexicon=LexiconDataSet()):
    """

    :param lexicon:
    :return:
    """

    # TODO paths
    lexicalUnitDir = "data/fndata-1.5-with-dev/train/lu"
    lexicalFrameDir = "data/fndata-1.5-with-dev/frame"
    lexicon.createLexicon(lexicalUnitDir)
    lexicon.createFrameToFE(lexicalFrameDir)

    lexicon.loaded = True
    lexicon.save("data/lexicon.file")


def create_salsa_lexicon(lexicon=LexiconDataSet()):
    """

    :param lexicon:
    :return:
    """

    # NOTE: Salsa has no ĺexicalunit lexicon
    lexicalFrameDir = "data/raw/salsa/salsa-frames-2.0.xml"
    lexicon.createFrameToFESalsa(lexicalFrameDir)

    lexicon.loaded = True
    lexicon.save("lexiconSalsa.file")


def load_lexicon(lexiconPath):
    """

    :param lexiconPath:
    :return:
    """

    with lzma.open(lexiconPath, "rb") as f:
        lexicon = pickle.load(f)

    return lexicon


if __name__ == "__main__":
    create_lexicon()

    create_salsa_lexicon()
