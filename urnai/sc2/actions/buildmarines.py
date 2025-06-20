import random

from pysc2.env import sc2_env
from pysc2.lib import units

from urnai.sc2.actions import sc2_actions_aux as scaux
from urnai.sc2.actions.collectables import CollectablesActionSpace
from urnai.sc2.actions.sc2_actions import raw_functions_classes as sc2_actions


class BuildMarinesActionSpace(CollectablesActionSpace):
    BOTTOM_RIGHT_SUPPLY_DEPOT_X = 42
    BOTTOM_RIGHT_SUPPLY_DEPOT_Y = 43
    UPPER_RIGHT_BARRACK_X = 41
    UPPER_RIGHT_BARRACK_Y = 29

    MAP_PLAYER_SUPPLY_DEPOT_COORDINATES = [
        {'x': BOTTOM_RIGHT_SUPPLY_DEPOT_X, 'y': BOTTOM_RIGHT_SUPPLY_DEPOT_Y},
        {'x': BOTTOM_RIGHT_SUPPLY_DEPOT_X - 2, 'y': BOTTOM_RIGHT_SUPPLY_DEPOT_Y},
        {'x': BOTTOM_RIGHT_SUPPLY_DEPOT_X - 4, 'y': BOTTOM_RIGHT_SUPPLY_DEPOT_Y},
        {'x': BOTTOM_RIGHT_SUPPLY_DEPOT_X - 6, 'y': BOTTOM_RIGHT_SUPPLY_DEPOT_Y},
        {'x': BOTTOM_RIGHT_SUPPLY_DEPOT_X - 8, 'y': BOTTOM_RIGHT_SUPPLY_DEPOT_Y},
        {'x': BOTTOM_RIGHT_SUPPLY_DEPOT_X - 10, 'y': BOTTOM_RIGHT_SUPPLY_DEPOT_Y},
        {'x': BOTTOM_RIGHT_SUPPLY_DEPOT_X - 12, 'y': BOTTOM_RIGHT_SUPPLY_DEPOT_Y},
        {'x': BOTTOM_RIGHT_SUPPLY_DEPOT_X - 14, 'y': BOTTOM_RIGHT_SUPPLY_DEPOT_Y},
        {'x': BOTTOM_RIGHT_SUPPLY_DEPOT_X - 5, 'y': BOTTOM_RIGHT_SUPPLY_DEPOT_Y - 3},
        {'x': BOTTOM_RIGHT_SUPPLY_DEPOT_X - 7, 'y': BOTTOM_RIGHT_SUPPLY_DEPOT_Y - 3},
        {'x': BOTTOM_RIGHT_SUPPLY_DEPOT_X - 9, 'y': BOTTOM_RIGHT_SUPPLY_DEPOT_Y - 3},
        {'x': BOTTOM_RIGHT_SUPPLY_DEPOT_X - 6, 'y': BOTTOM_RIGHT_SUPPLY_DEPOT_Y - 6},
        {'x': BOTTOM_RIGHT_SUPPLY_DEPOT_X - 8, 'y': BOTTOM_RIGHT_SUPPLY_DEPOT_Y - 6},
        {'x': BOTTOM_RIGHT_SUPPLY_DEPOT_X - 6, 'y': BOTTOM_RIGHT_SUPPLY_DEPOT_Y - 9},
        {'x': BOTTOM_RIGHT_SUPPLY_DEPOT_X - 8, 'y': BOTTOM_RIGHT_SUPPLY_DEPOT_Y - 9},
    ]

    MAP_PLAYER_BARRACK_COORDINATES = [
        {'x': UPPER_RIGHT_BARRACK_X, 'y': UPPER_RIGHT_BARRACK_Y},
        {'x': UPPER_RIGHT_BARRACK_X, 'y': UPPER_RIGHT_BARRACK_Y + 3},
        {'x': UPPER_RIGHT_BARRACK_X, 'y': UPPER_RIGHT_BARRACK_Y + 6},
        {'x': UPPER_RIGHT_BARRACK_X, 'y': UPPER_RIGHT_BARRACK_Y + 9},
        {'x': UPPER_RIGHT_BARRACK_X - 4, 'y': UPPER_RIGHT_BARRACK_Y},
        {'x': UPPER_RIGHT_BARRACK_X - 7, 'y': UPPER_RIGHT_BARRACK_Y},
        {'x': UPPER_RIGHT_BARRACK_X - 11, 'y': UPPER_RIGHT_BARRACK_Y},
        {'x': UPPER_RIGHT_BARRACK_X - 14, 'y': UPPER_RIGHT_BARRACK_Y},
    ]

    def __init__(self):
        super().__init__()

        self.do_nothing = "Collect"
        self.build_supply_depot = "Build_SupplyDepot_pt"
        self.build_barrack = "Build_Barracks_pt"
        self.build_marine = "BuildMarine"
        self.actions = {0: self.do_nothing,
                        1: self.build_supply_depot, 
                        2: self.build_barrack, 
                        3: self.build_marine}
        self.named_actions = ['do_nothing', 'build_supply_depot', 'build_barrack', 
                              'build_marine']
        self.action_indices = range(len(self.actions))
        self.barrack_coords = \
            BuildMarinesActionSpace.MAP_PLAYER_BARRACK_COORDINATES
        self.supply_depot_coords = \
            BuildMarinesActionSpace.MAP_PLAYER_SUPPLY_DEPOT_COORDINATES
    
    def solve_action(self, action_idx, obs):
        if action_idx is not None:
            if action_idx is not self.noaction:
                if action_idx not in self.actions:
                    raise ValueError(f"Invalid action index: {action_idx}. "+
                                     f"Valid actions: {list(self.actions.keys())}")
                action = self.actions[action_idx]
                if action == self.do_nothing:
                    self.collect_idle(obs)
                elif action == self.build_supply_depot:
                    coord = random.choice(self.supply_depot_coords)
                    self.build_pt(obs, coord, self.build_supply_depot)
                elif action == self.build_barrack:
                    coord = random.choice(self.barrack_coords)
                    self.build_pt(obs, coord, self.build_barrack)
                else:
                    self.build_marine_(obs)
        else:
            self.reset()

    def collect_idle(self, obs):
        scv = scaux.get_random_idle_worker(obs, sc2_env.Race.terran)
        if scv is not scaux._NO_UNITS:
            minerals = scaux.get_neutral_units_by_type(obs, units.Neutral.MineralField)
            if minerals:
                mineral = random.choice(minerals)
                self.pending_actions.append(
                    sc2_actions["Harvest_Gather_unit"].run('queued', scv.tag, 
                                                           mineral.tag))

    def select_random_scv(self, obs):
        # get SCV list
        scvs = scaux.get_units_by_type(obs, units.Terran.SCV)
        if not scvs:
            return None
        length = len(scvs)
        scv = scvs[random.randint(0, length - 1)]
        return scv
    
    def build_pt(self, obs, coord, build_action_pt):
        x, y = coord['x'], coord['y']
        scv = self.select_random_scv(obs)
        # append action to build building
        if scv is not None:
            self.pending_actions.append(
                sc2_actions[build_action_pt].run('now', scv.tag, [x, y]))

    def build_marine_(self, obs):
        barracks = scaux.get_units_by_type(obs, units.Terran.Barracks)
        if len(barracks) > 0:
            barrack = random.choice(barracks)
            self.pending_actions.append(
                sc2_actions["Train_Marine_quick"].run('now', barrack.tag))