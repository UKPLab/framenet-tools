from framenet_tools.data_handler.annotation import Annotation
from framenet_tools.config import ConfigManager


class RoleIdentifier(object):

    def __init__(self, cM: ConfigManager):
        self.cM = cM

    def predict_roles(self, annotation: Annotation):
        """
        Predict roles for all spans contained in the given annotation object

        NOTE: Manipulates the given annotation object!

        :param annotation: The annotation object to predict the roles for
        :return:
        """

        # TODO