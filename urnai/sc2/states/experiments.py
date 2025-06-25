from abc import abstractmethod
from enum import Enum
import math

import numpy as np

from urnai.states.state_base import StateBase


class StateType(Enum):
    STATE_MAP = 'map'
    STATE_NON_SPATIAL = 'non_spatial_only'
    STATE_BOTH = 'map_and_non_spatial'

STATE_MAP_DEFAULT_REDUCTIONFACTOR = 1
STATE_MAX_COLL_DIST = 15

class ExperimentsState(StateBase):

    def __init__(self, trim_map : bool = False, 
                 method : StateType = StateType.STATE_MAP,
                 map_size = (64, 64)):
        
        """
        map_reduction_factor's default value is 1.
        Example: if the value is 2, the map's size gets reduced by half.

        Non-spatial is state is composed of:
            . x distance to next mineral shard
            . y distance to next mineral shard

        """

        self.previous_state = None
        self.method = method
        self.map_size = map_size
        self.map_reduction_factor = STATE_MAP_DEFAULT_REDUCTIONFACTOR

        self.non_spatial_maximums = [
            STATE_MAX_COLL_DIST,
            STATE_MAX_COLL_DIST,
        ]

        self.non_spatial_minimums = [
            0,
            0,
        ]
        
        self.non_spatial_state = [
            0,
            0,
        ]

        self.trim_map = trim_map
        self.trim_factor = (22/64, 0.25)
        self.reset()

    def update(self, obs):
        state = []
        if self.method == StateType.STATE_MAP:
            state = self.build_map(obs)
        elif self.method == StateType.STATE_NON_SPATIAL:
            state = self.build_non_spatial_state(obs)
        elif self.method == StateType.STATE_BOTH:
            state = self.build_map(obs)
            state += self.build_non_spatial_state(obs)

        self._dimension = len(state)
        self._state = state

        return state

    def build_map(self, obs):
        map_ = self.build_basic_map(obs)
        map_ = self.reduce_map(map_)

        return map_

    @abstractmethod
    def build_basic_map(self, obs):
        ...

    def normalize_map(self, map_):
        return (map_ - map_.min()) / (map_.max() - map_.min())

    def normalize_non_spatial_list(self):
        for i in range(len(self.non_spatial_state)):
            value = self.non_spatial_state[i]
            max_ = self.non_spatial_maximums[i]
            min_ = self.non_spatial_minimums[i]
            value = self.normalize_value(value, max_, min_)
            self.non_spatial_state[i] = value

    def normalize_value(self, value, max_, min_=0):
        return (value - min_) / (max_ - min_)
    
    @property
    def dimension(self):
        if self.method == StateType.STATE_MAP:
            if self.trim_map:
                a = int(self.trim_factor[0] * 
                        self.map_size[0] / self.map_reduction_factor)
                b = int(self.trim_factor[1] * 
                        self.map_size[1] / self.map_reduction_factor)
            else:
                a = int(self.map_size[0] / self.map_reduction_factor)
                b = int(self.map_size[1] / self.map_reduction_factor)
            return int(a * b)
        elif self.method == StateType.STATE_NON_SPATIAL:
            return len(self.non_spatial_state)
    
    @property
    def state(self):
        return self._state

    def build_non_spatial_state(self, obs):
        return None
    
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
    
    def trim_matrix(self, matrix, x1, y1, x2, y2):
        """
        This function extracts a submatrix of a 
        2D numpy array.

        The arguments x1, y1 and x2, y2 are the
        top-left and bottom-right corners of
        this submatrix, respectively.

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
    
    def lower_featuremap_resolution(self, map, reduction_factor):
        """
        Reduces a matrix "resolution" by a reduction factor. If we have a 64x64 matrix 
        and rf=4 the map will be reduced to 16x16 in which every new element of the 
        matrix is an average from 4x4=16 elements from the original matrix
        """
        if reduction_factor == 1:
            return map

        N, M, Z = map.shape
        N = N // reduction_factor
        M = M // reduction_factor

        reduced_map = np.empty((N, M, Z))
        for i in range(N):
            for j in range(M):
                rf = reduction_factor
                reduced_map[i, j] = ((map[rf * i:rf * i + rf, rf * j:rf * j + rf].sum())
                                     / (rf * rf))

        return reduced_map