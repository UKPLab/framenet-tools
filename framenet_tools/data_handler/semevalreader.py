import xml.etree.ElementTree

from typing import List

from framenet_tools.config import ConfigManager
from framenet_tools.data_handler.annotation import Annotation
from framenet_tools.data_handler.reader import DataReader


def char_pos_to_sentence_pos(start_char: int, end_char: int, words: List[str]):
    """
    Converts positions of char spans in a sentence into word positions.

    NOTE: Returned end position is represented inclusive!

    :param start_char: The first character of the span
    :param end_char: The last character of the span
    :param words: A list of words in a sentence
    :return: The start and end position of the WORD in the sentence
    """

    start = -1
    end = -1

    chars = 0

    for i in range(len(words) + 1):
        if start == -1 and start_char <= chars:
            start = i

        if end == -1 and end_char < chars:
            return start, max(i - 1, start)

        if i == len(words):
            break

        # +1 due to empty spaces between words
        chars += len(words[i]) + 1

    raise Exception("Inconsistency: position not inside sentence!")


class SemevalReader(DataReader):
    """
    A reader for the Semeval format.

    Inherits from DataReader
    """

    def __init__(self, cM: ConfigManager, path_xml: str = None):
        DataReader.__init__(self, cM)

        self.path_xml = path_xml

    def read_data(self, path_xml: str = None):
        """
        Reads a xml file and parses it into the datareader format.

        NOTE: Applying this function removes the previous dataset content

        :param path_xml: The path of the xml file
        :return:
        """

        if path_xml is not None:
            self.path_xml = path_xml

        if self.path_xml is None:
            raise Exception("Found no xml-file to read!")

        if self.path_xml.rsplit(".")[-1] != "xml":
            raise Exception("File is not a xml-file!")

        tree = xml.etree.ElementTree.parse(self.path_xml)
        root = tree.getroot()

        self.digest_tree(root)

    def digest_tree(self, root: xml.etree.ElementTree):
        """
        Parses the xml-tree into a DataReader object.

        :param root: The root node of the tree
        :return:
        """

        sent_num = 0

        # Structure as define by semeval
        for sentences in root.findall(
            ".documents/document/paragraphs/paragraph/sentences/sentence"
        ):
            sentence = sentences.find("text").text

            raw_sent = sentence
            words = raw_sent.split(" ")

            while "" in words:
                words.remove("")
            self.sentences.append(words)

            for annotation in sentences.findall("annotationSets/annotationSet"):

                frame = annotation.get("frameName")

                data = annotation.findall("./layers/layer")
                fee_node = data[0].findall(".labels/label")

                start_char = int(fee_node[0].get("start"))
                end_char = int(fee_node[-1].get("end"))
                start, end = char_pos_to_sentence_pos(start_char, end_char, words)

                position = (start, end)

                fee = words[start]
                fee_raw = words[start]

                roles = []
                role_positions = []

                for role in data[1:]:
                    for labels in role:
                        for label in labels:
                            fe = label.get("name")
                            start_char = int(label.get("start"))
                            end_char = int(label.get("end"))
                            start, end = char_pos_to_sentence_pos(
                                start_char, end_char, words
                            )

                            roles.append(fe)
                            role_positions.append((start, end))

                if sent_num >= len(self.annotations):
                    self.annotations.append([])

                self.annotations[sent_num].append(
                    Annotation(
                        frame,
                        fee,
                        position,
                        fee_raw,
                        self.sentences[sent_num],
                        roles,
                        role_positions,
                    )
                )

            sent_num += 1
