from urnai.utils.constants import RTSGeneralization

from .findanddefeat import FindAndDefeatGeneralizedStatebuilder


class DefeatEnemiesGeneralizedStatebuilder(FindAndDefeatGeneralizedStatebuilder):

    def __init__(self, method=RTSGeneralization.STATE_MAP):
        super().__init__(method=method)
