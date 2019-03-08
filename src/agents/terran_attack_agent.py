import random
import math
import os.path
import numpy as np
from .base.sc2_abagent import SC2Agent
from pysc2.lib import actions, features
from pysc2.agents import base_agent

# Defining constants for our agent
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4

_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21

_NOT_QUEUED = [0]
_QUEUED = [1]



## Defining our agent's rewards
KILL_UNIT_REWARD = 0.2
KILL_BUILDING_REWARD = 0.5



# Defining our agent's class
class TerranAgent(SC2Agent):

    def __init__(self, action_wrapper):
        super(TerranAgent, self).__init__(action_wrapper)

        # Properties to track the change of values used in our reward system
        self.previous_killed_unit_score = 0
        self.previous_killed_building_score = 0

        self.previous_action = None
        self.previous_state = None


    def build_state(self, obs):
        ## Defining our state and calculating the reward
        unit_type = obs.observation.feature_screen[_UNIT_TYPE]

        # Whether or not our supply depot was built
        depot_y, depot_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
        supply_depot_count = 1 if depot_y.any() else 0

        # Whether or not our barracks were built
        barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
        barracks_count = 1 if barracks_y.any else 0

        # The supply limit
        supply_limit = obs.observation.player[4]
        # The army supply
        army_supply = obs.observation.player[5]

        # Getting values from the cumulative score system
        killed_unit_score = obs.observation.score_cumulative[5]
        killed_building_score = obs.observation.score_cumulative[6]

        # Defining our state, considering our enemies' positions.
        current_state = np.zeros(20)
        current_state[0] = supply_depot_count
        current_state[1] = barracks_count
        current_state[2] = supply_limit
        current_state[3] = army_supply

        # Insteading of making a vector for all coordnates on the map, we'll discretize our enemy space
        # and use a 16x16 grid to store enemy positions by marking a square as 1 if there's any enemy on it.
        hot_squares = np.zeros(16)
        enemy_y, enemy_x = (obs.observation.feature_minimap[_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
        for i in range(0, len(enemy_y)):
            y = int(math.ceil((enemy_y[i] + 1) / 16))
            x = int(math.ceil((enemy_x[i] + 1) / 16))

            hot_squares[((y - 1) * 4) + (x - 1)] = 1

        if not self.base_top_left:
            hot_squares = hot_squares[::-1]

        for i in range(0, 16):
            # Adds 4 to account for supply_depot_count, barracks_count, supply_limit and army_supply
            current_state[i + 4] = hot_squares[i]
        
        return current_state


    def get_state_dim(self):
        return [20]


    def step(self, obs):
        super(TerranAgent, self).step(obs)

        if obs.first():
            # Setting our base position
            player_y, player_x = (obs.observation.feature_minimap[_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
            self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0


        if obs.last():
            self.model.save()


        # Taking the first step for a smart action
        if self.action_wrapper.is_action_done():
            ## Building our agent's state
            current_state = self.build_state(obs)

            # Getting values from the cumulative score system
            killed_unit_score = obs.observation.score_cumulative[5]
            killed_building_score = obs.observation.score_cumulative[6]
            
            # If it's not the first step, we can learn
            if self.previous_action is not None:
                reward = 0

                if killed_unit_score > self.previous_killed_unit_score:
                    reward += KILL_UNIT_REWARD

                if killed_building_score > self.previous_killed_building_score:
                    reward += KILL_BUILDING_REWARD

                self.model.learn(self.previous_state, self.previous_action, reward, current_state, done=False)


            excluded_actions = self.action_wrapper.get_excluded_actions(obs)

            # Selects an action by passing the current state as a string. Our dataframe is indexed by strings.
            # We only select an action when move_number == 0, and we keep it until move_number == 2
            rl_action = self.model.choose_action(current_state, excluded_actions)

            # Saving the score system's current values
            self.previous_killed_unit_score = killed_unit_score
            self.previous_killed_building_score = killed_building_score
            self.previous_state = current_state
            self.previous_action = rl_action
            x = 0
            y = 0

        return self.action_wrapper.get_action(self.previous_action, obs)
