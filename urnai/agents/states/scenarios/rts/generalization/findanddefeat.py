from urnai.utils.constants import RTSGeneralization

from .collectables import CollectablesGeneralizedStatebuilder


class FindAndDefeatGeneralizedStatebuilder(CollectablesGeneralizedStatebuilder):

    def __init__(self, method=RTSGeneralization.STATE_MAP):
        super().__init__(method=method)

    def build_drts_map(self, obs):
        map_ = self.build_basic_drts_map(obs)
        map_ = self.normalize_map(map_)

        return map_
