import logging
import nltk
import spacy

from nltk.stem import WordNetLemmatizer
from nltk.tree import Tree
from typing import List


class PosTagger(object):
    """
    PosTagger provides options for assigning POS-tags to sentences.

    Either by spacy or nltk.
    """

    def __init__(self, use_spacy: bool):

        self.use_spacy = use_spacy

        if self.use_spacy:
            self.nlp = spacy.load("en_core_web_sm")
        else:
            self.lemmatizer = WordNetLemmatizer()

    def get_tags(self, sentence: List[str]):
        """
        Returns the POS-tags of a given sentence.

        :param sentence: The sentence, given as a list of words
        :return: A list of POS-tags
        """

        if self.use_spacy:
            return self.get_tags_spacy(sentence)

        return self.get_tags_nltk(sentence)

    def get_tags_spacy(self, tokens: List[str]):
        """
        The spacy version of the get_tags method

        :param tokens:The sentence, given as a list of words
        :return: A list of POS-tags
        """

        sentence = ""

        if len(tokens) > 0:
            sentence = tokens[0]

        for token in tokens[1:]:
            sentence += " " + token

        doc = self.nlp(sentence)
        pData = []

        for token in doc:
            text = token.text
            ne = token.ent_type_
            tag = token.tag_
            lemma = token.lemma_

            logging.debug(f"Token: {text}, NE: {ne}, Tag: {tag}, Lemma: {lemma}")

            if ne == "":
                ne = "-"

            pData.append([text, tag, lemma, ne])

        return pData

    def get_tags_nltk(self, tokens: List[str]):
        """
        Gets lemma, pos and NE for each token

        :param tokens: A list of tokens from a sentence
        :return: A 2d-Array containing lemma, pos and NE for each token
        """

        pos_tags = []
        lemmas = []
        nes = []

        tags = nltk.pos_tag(tokens)

        for tag in tags:
            pos_tags.append(tag[1])
            lemmas.append(
                self.lemmatizer.lemmatize(tag[0], pos=get_pos_constants(tag[1]))
            )
        chunks = nltk.ne_chunk(tags)
        for chunk in chunks:
            if isinstance(chunk, Tree):
                nes.append(chunk.label())
            else:
                nes.append("-")
        pData = []

        for t, p, l, n in zip(tokens, pos_tags, lemmas, nes):
            pData.append([t, p, l, n])

        return pData


def get_pos_constants(tag: str):
    """
    Static function for tag conversion

    :param tag: The given pos tag
    :return: The corresponding letter
    """

    if tag.startswith("J"):
        return "a"
    elif tag.startswith("V"):
        return "v"
    elif tag.startswith("N"):
        return "n"
    elif tag.startswith("R"):
        return "r"
    else:
        return "n"
