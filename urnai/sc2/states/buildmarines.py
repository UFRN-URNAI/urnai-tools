import numpy as np
from pysc2.lib import units as sc2units

import urnai.sc2.actions.sc2_actions_aux as sc2aux
from urnai.sc2.actions.buildmarines import BuildMarinesActionSpace
from urnai.states.state_base import StateBase

STATE_MAXIMUM_MINERALS = 10000
INITIAL_NUMBER_OF_SCV = 12
INITIAL_NUMBER_OF_SUPPLY = 15
MAXIMUM_NUMBER_OF_SUPPLY_DEPOT = len(BuildMarinesActionSpace. \
                                     MAP_PLAYER_SUPPLY_DEPOT_COORDINATES)
MAXIMUM_NUMBER_OF_BARRACKS = len(BuildMarinesActionSpace. \
                                 MAP_PLAYER_BARRACK_COORDINATES)
MAXIMUM_NUMBER_OF_MARINES = INITIAL_NUMBER_OF_SUPPLY - INITIAL_NUMBER_OF_SCV +\
                            8 * MAXIMUM_NUMBER_OF_SUPPLY_DEPOT


class BuildMarinesState(StateBase):

    def __init__(self):
        
        self.non_spatial_maximums = [
            STATE_MAXIMUM_MINERALS,
            MAXIMUM_NUMBER_OF_SUPPLY_DEPOT,
            MAXIMUM_NUMBER_OF_BARRACKS,
            MAXIMUM_NUMBER_OF_MARINES,
        ]
        self.non_spatial_minimums = [0, 0, 0, 0]
        self.non_spatial_state = [0, 0, 0, 0]

        self.reset()

    def update(self, obs):
        state = []
        state = self.build_non_spatial_state(obs)

        self._dimension = len(state)
        self._state = state
        self.non_spatial_state = np.array(self.non_spatial_state)

        return state

    def normalize_non_spatial_list(self):
        for i in range(len(self.non_spatial_state)):
            value = self.non_spatial_state[i]
            max_ = self.non_spatial_maximums[i]
            min_ = self.non_spatial_minimums[i]
            value = self.normalize_value(value, max_, min_)
            self.non_spatial_state[i] = value
    
    def normalize_value(self, value, max_, min_=0):
        return (value - min_) / (max_ - min_)
    
    
    def build_non_spatial_state(self, obs):
        self.non_spatial_state[0] = obs.player.minerals
        self.non_spatial_state[1] = sc2aux.get_my_units_amount(
            obs, sc2units.Terran.SupplyDepot)
        self.non_spatial_state[2] = sc2aux.get_my_units_amount(
            obs, sc2units.Terran.Barracks)
        self.non_spatial_state[3] = sc2aux.get_my_units_amount(
            obs, sc2units.Terran.Marine)
        self.normalize_non_spatial_list()
        return self.non_spatial_state

    @property
    def dimension(self):
        return len(self.non_spatial_state)
    
    @property
    def state(self):
        return self._state

    def reset(self):
        self._state = None
        self.non_spatial_state = [0, 0, 0, 0]