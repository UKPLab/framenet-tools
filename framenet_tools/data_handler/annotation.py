
class Annotation(object):
    def __init__(
        self,
        frame: str = "Default",
        fee: str = None,
        position: int = None,
        fee_raw: str = None,
        sentence: list = None,
        roles: str = None,
        role_positions: list = None
    ):
        self.frame = frame
        self.fee = fee
        self.position = position
        self.fee_raw = fee_raw
        self.sentence = sentence
        self.roles = roles
        self.role_positions = role_positions

    def create_handle(self):
        return [self.frame, self.fee, self.position, self.fee_raw, self.sentence, self.roles, self.role_positions]

    def __eq__(self, x):
        equal = True

        for h1, h2 in zip(self.create_handle(), x.create_handle()):
            equal &= h1 == h2

        return equal
