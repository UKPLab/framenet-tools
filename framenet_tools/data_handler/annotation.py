from typing import List, Tuple


class Annotation(object):
    """
    Annotation class

    Saves and manages all data of one frame for a given sentence.
    """

    def __init__(
        self,
        frame: str = "Default",
        fee: str = None,
        position: int = None,
        fee_raw: str = None,
        sentence: List[str] = [],
        roles: List[str] = [],
        role_positions: List[Tuple[int, int]] = []
    ):
        self.frame = frame
        self.fee = fee
        self.position = position
        self.fee_raw = fee_raw
        self.sentence = sentence
        self.roles = roles
        self.role_positions = role_positions
        self.embedded_frame = None
        self.frame_confidence = [[frame, 1.0]]

    def create_handle(self):
        """
        Helper function for ease of programmatic comparison

        NOTE: FEE is not compared due to possible differences during preprocessing!

        :return: A handle consisting of all data saved in this object
        """
        return [self.frame, self.position, self.fee_raw, self.sentence, self.roles, self.role_positions]

    def __eq__(self, x):
        """
        The overwriting of the comparison function

        :param x: Another instance of this class
        :return: True if equal, otherwise false
        """
        equal = True

        for h1, h2 in zip(self.create_handle(), x.create_handle()):
            equal &= h1 == h2

        return equal
