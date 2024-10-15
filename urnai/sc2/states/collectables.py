import math
from statistics import mean

import numpy as np
from pysc2.lib import units as sc2units

import urnai.sc2.actions.sc2_actions_aux as sc2aux
from urnai.states.state_base import StateBase

# from urnai.utils.constants import Games, RTSGeneralization

STATE_MAP = 'map'
STATE_NON_SPATIAL = 'non_spatial_only'
STATE_BOTH = 'map_and_non_spatial'
STATE_MAP_DEFAULT_REDUCTIONFACTOR = 1
STATE_MAX_COLL_DIST = 15


class CollectablesState(StateBase):

    def __init__(self, trim_map=False):
        self.previous_state = None
        self.method = STATE_MAP
        # number of quadrants is the amount of parts
        # the map should be reduced
        # this helps the agent to
        # deal with the big size
        # of state space
        # if -1 (default value), the map
        # wont be reduced
        self.map_reduction_factor = STATE_MAP_DEFAULT_REDUCTIONFACTOR
        self.non_spatial_maximums = [
            STATE_MAX_COLL_DIST,
            STATE_MAX_COLL_DIST,
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
        self.reset()

    def update(self, obs):
        state = []
        if self.method == STATE_MAP:
            state = self.build_map(obs)
        elif self.method == STATE_NON_SPATIAL:
            state = self.build_non_spatial_state(obs)
        elif self.method == STATE_BOTH:
            state = self.build_map(obs)
            state += self.build_non_spatial_state(obs)

        state = np.asarray(state).flatten()
        self._dimension = len(state)
        # state = state.reshape((1, len(state)))
        self._state = state

        return state

    def build_map(self, obs):
        map_ = self.build_basic_map(obs)
        map_ = self.reduce_map(map_)
        map_ = self.normalize_map(map_)

        return map_

    def build_basic_map(self, obs):

        map_ = np.zeros(obs.feature_minimap[0].shape)
        marines = sc2aux.get_units_by_type(obs, sc2units.Terran.Marine)
        shards = sc2aux.get_all_neutral_units(obs)

        for marine in marines:
            map_[marine.y][marine.x] = 7

        for shard in shards:
            map_[shard.y][shard.x] = 100

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
    
    # TODO: Remove magic numbers
    @property
    def dimension(self):
        if self.method == STATE_MAP:
            if self.trim_map:
                a = int(22 / self.map_reduction_factor)
                b = int(16 / self.map_reduction_factor)
            else:
                a = int(64 / self.map_reduction_factor)
                b = int(64 / self.map_reduction_factor)
            return int(a * b)
        elif self.method == STATE_NON_SPATIAL:
            return len(self.non_spatial_state)
    
    @property
    def state(self):
        return self._state

    def get_marine_mean(self, obs):
        xs = []
        ys = []

        for unit in sc2aux.get_units_by_type(obs, sc2units.Terran.Marine):
            xs.append(unit.x)
            ys.append(unit.y)

        x_mean = mean(xs)
        y_mean = mean(ys)

        return x_mean, y_mean

    def get_closest_mineral_shard_x_y(self, obs):
        closest_distance = STATE_MAX_COLL_DIST
        x, y = self.get_marine_mean(obs)
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

    def build_non_spatial_state(self, obs):
        x, y = self.get_closest_mineral_shard_x_y(obs)
        # position 0: distance x to closest shard
        self.non_spatial_state[0] = int(x)
        # position 1: distance y to closest shard
        self.non_spatial_state[1] = int(y)
        # position 2: number of remaining shards
        #        self.non_spatial_state[2]=np.count_nonzero(obs.feature_minimap[4]==16)
        self.normalize_non_spatial_list()
        return self.non_spatial_state

    def calculate_distance(self, x1, y1, x2, y2):
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return dist

    def reduce_map(self, map_):
        if self.trim_map:
            x1, y1 = 22, 28
            x2, y2 = 43, 43
            map_ = self.trim_matrix(map_, x1, y1, x2, y2)
        return self.lower_featuremap_resolution(map_, self.map_reduction_factor)
    
    def reset(self):
        self._state = None
        self._dimension = None
    
    def trim_matrix(matrix, x1, y1, x2, y2):
        """
        If you have a 2D numpy array
        and you want a submatrix of that array,
        you can use this function to extract it.
        You just need to tell this function
        what are the top-left and bottom-right
        corners of this submatrix, by setting
        x1, y1 and x2, y2.
        For example: some maps of StarCraft II
        have parts that are not walkable, this
        happens specially in PySC2 mini-games
        where only a small portion of the map
        is walkable. So, you may want to trim
        this big map (generally a 64x64 matrix)
        and leave only the useful parts.
        """
        matrix = np.delete(matrix, np.s_[0:x1:1], 1)
        matrix = np.delete(matrix, np.s_[0:y1:1], 0)
        matrix = np.delete(matrix, np.s_[x2 - x1 + 1::1], 1)
        matrix = np.delete(matrix, np.s_[y2 - y1 + 1::1], 0)
        return matrix
    
    def lower_featuremap_resolution(self, map, rf):  # rf = reduction_factor
        """
        Reduces a matrix "resolution" by a reduction factor. If we have a 64x64 matrix 
        and rf=4 the map will be reduced to 16x16 in which every new element of the 
        matrix is an average from 4x4=16 elements from the original matrix
        """
        if rf == 1:
            return map

        N, M = map.shape
        N = N // rf
        M = M // rf

        reduced_map = np.empty((N, M))
        for i in range(N):
            for j in range(M):
                # reduction_array = map[rf*i:rf*i+rf, rf*j:rf*j+rf].flatten()
                # reduced_map[i,j]  = Counter(reduction_array).most_common(1)[0][0]

                reduced_map[i, j] = ((map[rf * i:rf * i + rf, rf * j:rf * j + rf].sum())
                                     / (rf * rf))

        return reduced_map