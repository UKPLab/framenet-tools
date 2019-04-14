from typing import List

from framenet_tools.config import ConfigManager
from framenet_tools.data_handler.annotation import Annotation
from framenet_tools.data_handler.reader import DataReader
from framenet_tools.fee_identification.feeidentifier import FeeIdentifier
from framenet_tools.pipelinestage import PipelineStage


class FeeID(PipelineStage):

    def __init__(self, cM: ConfigManager):
        super().__init__(cM)

    def train(self, data: List[str]):

        # Nothing to train on this stage

        return

    def predict(self, m_reader: DataReader):

        # m_reader.predict_fees()

        annotations = []
        fee_finder = FeeIdentifier(self.cM)

        for sentence in m_reader.sentences:
            possible_fees = fee_finder.query([sentence])
            predicted_annotations = []

            # Create new Annotation for each possible frame evoking element
            for possible_fee in possible_fees:
                predicted_annotations.append(
                    Annotation(fee_raw=possible_fee, sentence=sentence)
                )

            annotations.append(predicted_annotations)

        m_reader.annotations = annotations

        """
        annotations = []

        for annotation in data:
            sentence = annotation.sentence

            possible_fees = self.fee_finder.query([sentence])

            predicted_annotations = []

            # Create new Annotation for each possible frame evoking element
            for possible_fee in possible_fees:
                predicted_annotations.append(
                    Annotation(fee_raw=possible_fee, sentence=sentence)
                )

            annotations.append(predicted_annotations)

        data = annotations
        """