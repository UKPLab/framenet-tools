import en_core_web_sm

from framenet_tools.config import ConfigManager
from framenet_tools.data_handler.annotation import Annotation

class SpanIdentifier(object):

    def __init__(self, cM: ConfigManager):
        self.cM = cM

        self.nlp = en_core_web_sm.load()

    def query(self, annotation: Annotation):
        """

        :param annotation:
        :return:
        """

        tokens = annotation.sentence

        possible_roles = []
        sentence = ""

        if len(tokens) > 0:
            sentence = tokens[0]

        for token in tokens:
            sentence += " " + token

        # doc = self.nlp(sentence)

        # TODO
        possible_roles.append((1, 1))

        return possible_roles

    def evaluate(self, prediction, role_positions):
        """

        :param prediction:
        :param role_positions:
        :return:
        """

        #if

        return None