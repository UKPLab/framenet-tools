import sys
import spacy

import en_core_web_sm

from copy import deepcopy

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

        '''
        # Warning: Get way to large, way to fast...
        for i in range(len(sentence)):
            for j in range(i, len(sentence)):

                possible_roles.append((i, j))
        '''

        #sentence ="Autonomous cars shift insurance liability toward manufacturers"

        doc = self.nlp(sentence)

        '''
        for token in doc:
           
            min_index = sys.maxsize
            max_index = -1

            for child in token.children:

                position = list(doc).index(child)

                if position < min_index:
                    min_index = position

                if position > max_index:
                    max_index = position

            if max_index != -1: #and min != sys.maxsize:
                span = (min(min_index, token.i), max(max_index, token.i))
            else:
                span = ((token.i, token.i))

            possible_roles.append(span)


        #print(possible_roles)
        #exit()
        '''

        root = [token for token in doc if token.head == token][0]
        #print(root)

        combinations = self.traverse_syntax_tree(root)

        for combination in combinations:
            t = (min(combination), max(combination))
            if t not in possible_roles:
                possible_roles.append(t)

        #print(possible_roles)

        #exit()

        return possible_roles

    def traverse_syntax_tree(self, node: spacy.tokens.Token):
        spans = []
        retrieved_spans = []

        left_nodes = list(node.lefts)
        right_nodes = list(node.rights)

        for x in left_nodes:
            subs = self.traverse_syntax_tree(x)
            retrieved_spans += subs
            #spans.append([sub.append(node.i) for sub in subs])

        for x in right_nodes:
            subs = self.traverse_syntax_tree(x)
            retrieved_spans += subs
            #spans.append([sub.append(node.i) for sub in subs])

        # print(spans)
        # retrieved_spans = deepcopy(spans)
        for span in retrieved_spans:
            spans.append(span)
            spans.append(span + [node.i])


        if not spans:
            spans.append([node.i])

        #print(spans)
        return spans

    def evaluate(self, prediction, role_positions):
        """

        :param prediction:
        :param role_positions:
        :return:
        """

        return None
