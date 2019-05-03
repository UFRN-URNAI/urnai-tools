import random
import numpy as np
from .base.abwrapper import ActionWrapper
from agents.actions.sc2 import *
from utils.agent_utils import one_hot_encode, transformDistance, transformLocation
from pysc2.lib import features


## Defining action constants. These are names of the actions our agent will try to use.
ACTION_DO_NOTHING = 'donothing'                 # The agent does nothing for 3 steps
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'  # Selects SCV > builds supply depot > send SCV to harvest minerals
ACTION_BUILD_BARRACKS = 'buildbarracks'         # Selects SCV > builds barracks > sends SCV to harvest minerals
ACTION_BUILD_MARINE = 'buildmarine'             # Selects all barracks > trains marines > nothing
ACTION_ATTACK = 'attack'                        # Selects army > attacks coordinates > nothing

_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index

_PLAYER_SELF = 1
_TERRAN_COMMANDCENTER = 18
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21
_TERRAN_SCV = 45


class SC2Wrapper(ActionWrapper):

    def __init__(self):
        self.move_number = 0

        '''
        We're defining names for our actions for two reasons:
        1) Abstraction: By defining names for our actions as strings we can pour in extra info. EX: The ACTION_ATTACK_x_y action contains
        can be parsed to retrieve (x, y) coordinates and pass them to the actual PySC2 action.
         
        2) Readability: Using names instead of literal numbers makes it easier to tell which action is which.
        '''
        self.named_actions = [
            ACTION_DO_NOTHING,
            ACTION_BUILD_SUPPLY_DEPOT,
            ACTION_BUILD_BARRACKS,
            ACTION_BUILD_MARINE
        ]

        '''
        We're splitting the minimap into a 4x4 grid because the marine's effective range is able to cover
        the entire map from just this number of cells. For each (x, y) grid cell, we're defining an action called
        ACTION_ATTACK_x_y. When this actions is selected, we parse this string to retrieve this coordinate info
        and pass it as a parameter to the actual PySC2 action.
        '''
        for mm_x in range(0, 64):
            for mm_y in range(0, 64):
                if (mm_x + 1) % 32 == 0 and (mm_y + 1) % 32 == 0:
                    self.named_actions.append(ACTION_ATTACK + '_' + str(mm_x - 16) + '_' + str(mm_y - 16))
   
        '''
        In URNAI, the models can only return action indices. This index is passed by the agent to an ActionWrapper like this
        so that it decides what to do with the current action. For a complicated game like StarCraft 2 we can't just return an action,
        because most of them require extra parameters. So the get_action action of this wrapper is responsible for:
        1) Receiving an action index from the agent
        2) Selecting a named_action from the actions by using this index.
        3) Returning the PySC2 action that is equivalent to this named_action

        EX: 
        0) Agent receives action index 3 from its model
        1) Agent calls action_wrapper.get_action(3)
        2) get_action selects ACTION_BUILD_BARRACKS from its set of named actions
        3) get_action returns select_random_scv()
        '''
        self.action_indices = [idx for idx in range(len(self.named_actions))]


    def is_action_done(self):
        return self.move_number == 0

    
    def reset(self):
        self.move_number = 0


    def get_actions(self):
        return self.action_indices

    
    def split_action(self, smart_action):
        '''Breaks out (x, y) coordinates from a named action, if there are any.'''
        x = 0
        y = 0
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')

        return (smart_action, x, y)


    def get_excluded_actions(self, obs):
        unit_type = obs.feature_screen[_UNIT_TYPE]

        depot_y, depot_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
        supply_depot_count = int(round(len(depot_y) / 69))

        barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
        barracks_count = int(round(len(barracks_y) / 137))

        supply_used = obs.player[3]
        supply_limit = obs.player[4]
        supply_free = supply_limit - supply_used

        army_supply = obs.player[5]
        worker_supply = obs.player[6]

        # Adding invalid actions to the list of excluded actions
        excluded_actions = []
        # If the supply depot limit of 2 was reached, removes the ability to build it.
        if supply_depot_count == 2 or worker_supply == 0:
            excluded_actions.append(self.action_indices[1])
        # If we have no supply depots or we have 2 barracks, we remove the ability to build it.
        if supply_depot_count == 0 or barracks_count == 2 or worker_supply == 0:
            excluded_actions.append(self.action_indices[2])
        # If we don't have any barracks or have reached supply limit, remove the ability to train marines
        if supply_free == 0 or barracks_count == 0:
            excluded_actions.append(self.action_indices[3])
        # If we have no marines, we remove attack actions
        if army_supply == 0:
            excluded_actions.append(self.action_indices[4])
            excluded_actions.append(self.action_indices[5])
            excluded_actions.append(self.action_indices[6])
            excluded_actions.append(self.action_indices[7])
        
        return excluded_actions


    def get_action(self, action_idx, obs):
        named_action = self.named_actions[action_idx]
        named_action, x, y = self.split_action(named_action)
        

        # Initializing variables that are used to select the actions
        unit_type = obs.feature_screen[_UNIT_TYPE]
        player_y, player_x = (obs.feature_minimap[_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
        base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0
        player_cc_y, player_cc_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

        if self.move_number == 0:
            self.move_number += 1

            # Selects a random SCV, this is the first step to building a supply depot or barracks
            if named_action == ACTION_BUILD_BARRACKS or named_action == ACTION_BUILD_SUPPLY_DEPOT:
                return select_random_unit(obs)

            # Selects all barracks on the screen simultaneously
            elif named_action == ACTION_BUILD_MARINE:
                return select_all_barracks(obs)

            elif named_action == ACTION_ATTACK:
                return select_army(obs)

        elif self.move_number == 1:
            self.move_number += 1

            # Commands the SCV to build the depot at a given location. The place
            # we use to build supply depots and barracks are hard coded.
            if named_action == ACTION_BUILD_SUPPLY_DEPOT:
                # Calculates the number of supply depots currently built
                unit_type = obs.feature_screen[_UNIT_TYPE]
                depot_y, depot_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
                supply_depot_count = int(round(len(depot_y) / 69))

                if supply_depot_count < 2:
                    if player_cc_y.any():
                        # Builds supply depots at a fixed location
                        if supply_depot_count == 0:
                            target = transformDistance(round(player_cc_x.mean()), -35, round(player_cc_y.mean()), 0, base_top_left)
                        elif supply_depot_count == 1:
                            target = transformDistance(round(player_cc_x.mean()), -25, round(player_cc_y.mean()), -25, base_top_left)
                        return build_supply_depot(obs, target) 

            # Commands the selected SCV to build barracks at a given location
            elif named_action == ACTION_BUILD_BARRACKS:
                # Calculates the number of barracks currently built
                unit_type = obs.feature_screen[_UNIT_TYPE]
                barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
                barracks_count = int(round(len(barracks_y) / 137))

                if barracks_count < 2:
                    if player_cc_y.any():
                        # Builds barracks at a fixed location.
                        if barracks_count == 0:
                            target = transformDistance(round(player_cc_x.mean()), 15, round(player_cc_y.mean()), -9, base_top_left)
                        elif barracks_count == 1:
                            target = transformDistance(round(player_cc_x.mean()), 15, round(player_cc_y.mean()), 12, base_top_left)
                        return build_barracks(obs, target)

            # Tells the barracks to train a marine
            elif named_action == ACTION_BUILD_MARINE:
                return train_marine(obs)
            
            # Tells the agent to attack a location on the map
            elif named_action == ACTION_ATTACK:
                do_it = True
                
                # Checks if any SCV is selected. If so, the agent doesn't attack.
                if len(obs.single_select) > 0 and obs.single_select[0][0] == _TERRAN_SCV:
                    do_it = False
                if len(obs.multi_select) > 0 and obs.multi_select[0][0] == _TERRAN_SCV:
                    do_it = False
                if do_it:
                    x_offset = random.randint(-1, 1)
                    y_offset = random.randint(-1, 1)
                    target = transformLocation(int(x) + (x_offset * 8), int(y) + (y_offset * 8), base_top_left)
                    return attack_target_point(obs, target)

        elif self.move_number == 2:
            self.move_number = 0

            # Sends the SCV back to a mineral patch after it finished building.
            if named_action == ACTION_BUILD_BARRACKS or named_action == ACTION_BUILD_SUPPLY_DEPOT:
                return harvest_point(obs)

        return no_op()
