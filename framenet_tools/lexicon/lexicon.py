import xml.etree.ElementTree as ET
import os
import logging
import lzma
import pickle

FN = '{http://framenet.icsi.berkeley.edu}'


class LexiconDataSet(object):

    def __init__(self):
        self.loaded = False
        self.lexicon = dict()
        self.lexiconWithoutSyntax = dict()
        self.frameToFE = dict()

    def loadFileToLexicon(self, filePath):
        """

        :param filePath:
        :return:
        """

        tree = ET.parse(filePath)
        root = tree.getroot()

        lexicalUnit = root.get('name')
        lexicalUnitWithoutSyntax = lexicalUnit.rsplit('.')[0]

        frame = root.get('frame')

        for FrameElement in root.findall(FN + 'header/' + FN + 'frame/' + FN + 'FE'):
            if lexicalUnit not in self.lexicon:
                self.lexicon[lexicalUnit] = []

            if lexicalUnitWithoutSyntax not in self.lexiconWithoutSyntax:
                self.lexiconWithoutSyntax[lexicalUnitWithoutSyntax] = []

            self.lexicon[lexicalUnit].append([frame, FrameElement.get('name')])
            self.lexiconWithoutSyntax[lexicalUnitWithoutSyntax].append([frame, FrameElement.get('name')])

    def loadFrame(self, filePath):
        """

        :param filePath:
        :return:
        """

        tree = ET.parse(filePath)
        root = tree.getroot()

        lexicalUnit = root.get('name')

        for FrameElement in root.findall(FN + 'FE'):
            if lexicalUnit not in self.frameToFE:
                self.frameToFE[lexicalUnit] = []

            self.frameToFE[lexicalUnit].append([FrameElement.get('coreType'), FrameElement.get('name')])

    def createFrameToFE(self, framToFEDir):
        """

        :param framToFEDir:
        :return:
        """

        for file in os.listdir(framToFEDir):
            if file.endswith(".xml"):
                fullPath = os.path.join(framToFEDir, file)
                print(fullPath)
                self.loadFrame(fullPath)

    def loadSalsa(self, filePath):
        """

        :param filePath:
        :return:
        """

        tree = ET.parse(filePath)
        root = tree.getroot()

        for frame in root.findall('frame'):

            lexicalUnit = frame.get('name')
            print(frame.get('name'))
            for fes in frame.findall('fes'):
                for FrameElement in fes.findall('fe'):
                    if lexicalUnit not in self.frameToFE:
                        self.frameToFE[lexicalUnit] = []
                    self.frameToFE[lexicalUnit].append([FrameElement.get('coreType'), FrameElement.get('name')])

    def createFrameToFESalsa(self, fullPath):
        """

        :param fullPath:
        :return:
        """

        logging.info(f'{fullPath}')
        self.loadSalsa(fullPath)

    def createLexicon(self, lexicalUnitDir):
        """

        :param lexicalUnitDir:
        :return:
        """

        for file in os.listdir(lexicalUnitDir):
            if file.endswith(".xml"):
                fullPath = os.path.join(lexicalUnitDir, file)
                logging.info(f'{fullPath}')
                self.loadFileToLexicon(fullPath)

    def save(self, lexiconFile):
        """

        :param lexiconFile:
        :return:
        """

        if self.loaded:
            logging.info(f'Saving dataset to {lexiconFile}')
            with lzma.open(lexiconFile, 'wb') as f:
                pickle.dump(self, f)
        else:
            logging.error(f'Dataset not loaded, call "build" method first!')

