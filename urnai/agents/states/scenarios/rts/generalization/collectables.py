import math
from statistics import mean

import numpy as np
from pysc2.lib import units as sc2units
import urnai.agents.actions.sc2 as sc2aux
from urnai.agents.states.abstate import StateBuilder
from urnai.utils.constants import Games, RTSGeneralization
from urnai.utils.image import lower_featuremap_resolution
from urnai.utils.numpy_utils import trim_matrix


class CollectablesGeneralizedStatebuilder(StateBuilder):

    def __init__(self, method=RTSGeneralization.STATE_MAP,
                 map_reduction_factor=RTSGeneralization.STATE_MAP_DEFAULT_REDUCTIONFACTOR,
                 trim_map=False):
        self.previous_state = None
        self.method = method
        # number of quadrants is the amount of parts
        # the map should be reduced
        # this helps the agent to
        # deal with the big size
        # of state space
        # if -1 (default value), the map
        # wont be reduced
        self.map_reduction_factor = map_reduction_factor
        self.non_spatial_maximums = [
            RTSGeneralization.STATE_MAX_COLL_DIST,
            RTSGeneralization.STATE_MAX_COLL_DIST,
            #                RTSGeneralization.STATE_MAXIMUM_NUMBER_OF_MINERAL_SHARDS,
        ]
        self.non_spatial_minimums = [
            0,
            0,
            #                0,
        ]
        # non-spatial is composed of
        # X distance to next mineral shard
        # Y distance to next mineral shard
        # number of mineral shards left
        self.non_spatial_state = [
            0,
            0,
            #                0,
        ]
        self.trim_map = trim_map

    def get_game(self, obs):
        try:
            # a = obs.feature_minimap
            return Games.SC2
        except AttributeError as ae:
            if 'feature_minimap' in str(ae):
                return Games.DRTS
            else:
                raise

    def build_state(self, obs):
        game = self.get_game(obs)
        if game == Games.DRTS:
            state = self.build_drts_state(obs)
        else:
            state = self.build_sc2_state(obs)

        return state

    def build_drts_state(self, obs):
        state = []
        if self.method == RTSGeneralization.STATE_MAP:
            state = self.build_drts_map(obs)
        elif self.method == RTSGeneralization.STATE_NON_SPATIAL:
            state = self.build_non_spatial_drts_state(obs)
        elif self.method == RTSGeneralization.STATE_BOTH:
            state = self.build_drts_map(obs)
            state += self.build_non_spatial_drts_state(obs)

        state = np.asarray(state).flatten()
        state = state.reshape((1, len(state)))

        return state

    def build_sc2_state(self, obs):
        state = []
        if self.method == RTSGeneralization.STATE_MAP:
            state = self.build_sc2_map(obs)
        elif self.method == RTSGeneralization.STATE_NON_SPATIAL:
            state = self.build_non_spatial_sc2_state(obs)
        elif self.method == RTSGeneralization.STATE_BOTH:
            state = self.build_sc2_map(obs)
            state += self.build_non_spatial_sc2_state(obs)

        state = np.asarray(state).flatten()
        state = state.reshape((1, len(state)))

        return state

    def build_sc2_map(self, obs):
        map_ = self.build_basic_sc2_map(obs)
        map_ = self.reduce_map(map_)
        map_ = self.normalize_map(map_)

        return map_

    def build_basic_sc2_map(self, obs):
        # old map generation code
        # layer 3 is base (2 walkable area, 0 not)
        # layer 4 is units (1 friendly, 2 enemy, 16 mineral shards, 3 neutral
        # for y in range(len(obs.feature_minimap[3])):
        #     for x in range(len(obs.feature_minimap[3][y])):
        #         if obs.feature_minimap[3][y][x] == 2: map_[y][x] = 270 #drts walkable area

        # for y in range(len(obs.feature_minimap[4])):
        #     for x in range(len(obs.feature_minimap[4][y])):
        #         if obs.feature_minimap[4][y][x] == 1: map_[y][x] = 1 #drts 1 is peasant, 7
        #         is archer, which one is needed for the current map
        #         elif obs.feature_minimap[4][y][x] == 2: map_[y][x] = 7 #drts 1 is peasant, 7
        #         is archer, which one is needed for the current map
        #         elif obs.feature_minimap[4][y][x] == 16: map_[y][x] = 100 #drts 1000 was
        #         chosen by me to represent virtual shards
        #         elif obs.feature_minimap[4][y][x] == 3: map_[y][x] = 102 #drts 102 is gold

        map_ = np.zeros(obs.feature_minimap[0].shape)
        marines = sc2aux.get_units_by_type(obs, sc2units.Terran.Marine)
        shards = sc2aux.get_all_neutral_units(obs)

        for marine in marines:
            map_[marine.y][marine.x] = 7

        for shard in shards:
            map_[shard.y][shard.x] = 100

        return map_

    def build_drts_map(self, obs):
        map_ = self.build_basic_drts_map(obs)

        for y in range(len(obs['collectables_map'])):
            for x in range(len(obs['collectables_map'][y])):
                if obs['collectables_map'][y][x] == 1:
                    map_[y][x] = 100
                if map_[y][x] != 100 and map_[y][x] != 7:
                    map_[y][x] = 0

        map_ = self.reduce_map(map_)
        map_ = self.normalize_map(map_)

        return map_

    def build_basic_drts_map(self, obs):
        width = obs['map'].map_width
        height = obs['map'].map_height
        map_ = np.zeros((width, height))
        for tile in obs['tiles']:
            if int(tile.get_type_id()) == 102:  # 102 is gold
                map_[tile.y, tile.x] = tile.get_type_id()

        for unit in obs['units']:
            if unit.tile is not None:
                map_[unit.tile.y, unit.tile.x] = int(unit.type)

        return map_

    def normalize_map(self, map_):
        # map = (map_ - map_.min()) / (map_.max() - map_.min())
        return map_

    def normalize_non_spatial_list(self):
        for i in range(len(self.non_spatial_state)):
            value = self.non_spatial_state[i]
            max_ = self.non_spatial_maximums[i]
            min_ = self.non_spatial_minimums[i]
            value = self.normalize_value(value, max_, min_)
            self.non_spatial_state[i] = value

    def normalize_value(self, value, max_, min_=0):
        return (value - min_) / (max_ - min_)

    def get_state_dim(self):
        if self.method == RTSGeneralization.STATE_MAP:
            if self.trim_map:
                a = int(22 / self.map_reduction_factor)
                b = int(16 / self.map_reduction_factor)
            else:
                a = int(64 / self.map_reduction_factor)
                b = int(64 / self.map_reduction_factor)
            return int(a * b)
        elif self.method == RTSGeneralization.STATE_NON_SPATIAL:
            return len(self.non_spatial_state)

    def get_sc2_marine_mean(self, obs):
        xs = []
        ys = []

        for unit in sc2aux.get_units_by_type(obs, sc2units.Terran.Marine):
            xs.append(unit.x)
            ys.append(unit.y)

        x_mean = mean(xs)
        y_mean = mean(ys)

        return x_mean, y_mean

    def get_closest_sc2_mineral_shard_x_y(self, obs):
        closest_distance = RTSGeneralization.STATE_MAX_COLL_DIST
        x, y = self.get_sc2_marine_mean(obs)
        x_closest_distance, y_closest_distance = -1, -1
        for mineral_shard in sc2aux.get_all_neutral_units(obs):
            mineral_shard_x = mineral_shard.x
            mineral_shard_y = mineral_shard.y
            dist = self.calculate_distance(x, y, mineral_shard_x, mineral_shard_y)
            if dist < closest_distance:
                closest_distance = dist
                x_closest_distance = x - mineral_shard_x
                y_closest_distance = y - mineral_shard_y

        return abs(x_closest_distance), abs(y_closest_distance)

    def build_non_spatial_sc2_state(self, obs):
        x, y = self.get_closest_sc2_mineral_shard_x_y(obs)
        # position 0: distance x to closest shard
        self.non_spatial_state[0] = int(x)
        # position 1: distance y to closest shard
        self.non_spatial_state[1] = int(y)
        # position 2: number of remaining shards
        #        self.non_spatial_state[2] = np.count_nonzero(obs.feature_minimap[4] == 16)
        self.normalize_non_spatial_list()
        return self.non_spatial_state

    def build_non_spatial_drts_state(self, obs):
        x, y = self.get_closest_drts_mineral_shard_x_y(obs)
        # position 0: distance x to closest shard
        self.non_spatial_state[0] = int(x)
        # position 1: distance y to closest shard
        self.non_spatial_state[1] = int(y)
        # position 4: number of remaining shards
        #        self.non_spatial_state[2] = np.count_nonzero(obs['collectables_map'] == 1)
        self.normalize_non_spatial_list()
        return self.non_spatial_state

    def get_closest_drts_mineral_shard_x_y(self, obs):
        closest_distance = RTSGeneralization.STATE_MAX_COLL_DIST
        x, y = self.get_drts_army_mean(obs)
        x_closest_distance, y_closest_distance = -1, -1
        for mineral_shard_y in range(len(obs['collectables_map'])):
            for mineral_shard_x in range(len(obs['collectables_map'][0])):
                if obs['collectables_map'][mineral_shard_y][mineral_shard_x] == 1:
                    dist = self.calculate_distance(x, y, mineral_shard_x, mineral_shard_y)
                    if dist < closest_distance:
                        closest_distance = dist
                        x_closest_distance = x - mineral_shard_x
                        y_closest_distance = y - mineral_shard_y

        return abs(x_closest_distance), abs(y_closest_distance)

    def calculate_distance(self, x1, y1, x2, y2):
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return dist

    def get_drts_army_mean(self, obs):
        xs = []
        ys = []

        for unit in self.get_drts_player_units(obs, 0):
            xs.append(unit.tile.x)
            ys.append(unit.tile.y)

        x_mean = mean(xs)
        y_mean = mean(ys)

        return x_mean, y_mean

    def get_drts_player_units(self, obs, player):
        units = []
        for unit in obs['units']:
            if unit.get_player() == obs['players'][player]:
                units.append(unit)

        return units

    def get_drts_player_specific_type_units(self, obs, player, unit_id):
        all_units = self.get_drts_player_units(obs, player)
        specific_units = []

        for unit in all_units:
            if int(unit.type) == unit_id:
                specific_units.append(unit)

        return specific_units

    def reduce_map(self, map_):
        if self.trim_map:
            x1, y1 = 22, 28
            x2, y2 = 43, 43
            map_ = trim_matrix(map_, x1, y1, x2, y2)
        return lower_featuremap_resolution(map_, self.map_reduction_factor)
