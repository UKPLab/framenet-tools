import logging
import nltk

required_resources = [
    ["taggers/", "averaged_perceptron_tagger"],
    ["tokenizers/", "punkt"],
    ["corpora/", "wordnet"],
    ["chunkers/", "maxent_ne_chunker"],
    ["corpora/", "words"]
]


def download_resources():
    """
    Checks if the required resources from nltk are installed, if not they are downloaded.

    :return:
    """

    logging.info("Checking nltk resources.")
    for resource in required_resources:
        try:
            nltk.data.find(resource[0] + resource[1])
        except LookupError:
            logging.info(f"Did not find {resource[1]}, downloading...")
            nltk.download(resource[1])
