import numpy as np

from pysc2.lib import units as sc2units, features

import urnai.sc2.actions.sc2_actions_aux as sc2aux

from .experiments import ExperimentsState, StateType


STATE_MAP_DEFAULT_REDUCTIONFACTOR = 1
STATE_MAX_COLL_DIST = 15

class DefeatRoachesState(ExperimentsState):

    def __init__(self, trim_map : bool = False, 
                 method : StateType = StateType.STATE_MAP,
                 map_size = (64, 64)):
        super().__init__(trim_map, method, map_size)

    def build_basic_map(self, obs):

        map_ = np.zeros(
            (obs.feature_minimap[0].shape[0],
            obs.feature_minimap[0].shape[1], 3), dtype=np.uint8)
        marines = sc2aux.get_units_by_type(obs, sc2units.Terran.Marine)
        roaches = sc2aux.get_units_by_type(obs, sc2units.Zerg.Roach,
                                        features.PlayerRelative.ENEMY)

        for marine in marines:
            map_[marine.y][marine.x] = (255, 0, 0)

        for roach in roaches:
            map_[roach.y][roach.x] = (0, 255, 0)
        
        return map_

    def reduce_map(self, map_):
        if self.trim_map:
            x1, y1 = 22, 20
            x2, y2 = 44, 36
            map_ = self.trim_matrix(map_, x1, y1, x2, y2)
        return self.lower_featuremap_resolution(map_, self.map_reduction_factor)