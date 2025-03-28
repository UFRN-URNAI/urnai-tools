import random

from pysc2.env import sc2_env
from pysc2.lib import units

from urnai.sc2.actions import sc2_actions_aux as scaux
from urnai.sc2.actions.collectables import CollectablesActionSpace
from urnai.sc2.actions.sc2_actions import raw_functions_classes as sc2_actions


class BuildMarinesActionSpace(CollectablesActionSpace):
    SUPPLY_DEPOT_X = 42
    SUPPLY_DEPOT_Y = 42
    BARRACK_X = 39
    BARRACK_Y = 36

    # ACTION_DO_NOTHING = 7
    # ACTION_BUILD_SUPPLY_DEPOT = 8
    # ACTION_BUILD_BARRACK = 9
    # ACTION_BUILD_MARINE = 10

    MAP_PLAYER_SUPPLY_DEPOT_COORDINATES = [
        {'x': SUPPLY_DEPOT_X, 'y': SUPPLY_DEPOT_Y},
        {'x': SUPPLY_DEPOT_X - 2, 'y': SUPPLY_DEPOT_Y},
        {'x': SUPPLY_DEPOT_X - 4, 'y': SUPPLY_DEPOT_Y},
        {'x': SUPPLY_DEPOT_X - 6, 'y': SUPPLY_DEPOT_Y},
        {'x': SUPPLY_DEPOT_X - 8, 'y': SUPPLY_DEPOT_Y},
        {'x': SUPPLY_DEPOT_X - 10, 'y': SUPPLY_DEPOT_Y},
        {'x': SUPPLY_DEPOT_X - 12, 'y': SUPPLY_DEPOT_Y},
    ]

    MAP_PLAYER_BARRACK_COORDINATES = [
        {'x': BARRACK_X, 'y': BARRACK_Y},
        {'x': BARRACK_X, 'y': BARRACK_Y - 6},
    ]

    def __init__(self):
        super().__init__()

        self.do_nothing = 7
        self.build_supply_depot = 8
        self.build_barrack = 9
        self.build_marine = 10
        self.actions = [self.do_nothing, self.build_supply_depot, self.build_barrack,
                        self.build_marine]
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
                action = self.actions[action_idx]
                if action == self.do_nothing:
                    self.collect_idle(obs)
                elif action == self.build_supply_depot:
                    self.build_supply_depot(obs)
                elif action == self.build_barrack:
                    self.build_barrack(obs)
                elif action == self.build_marine:
                    self.build_marine(obs)
        else:
            self.reset()

    def collect_idle(self, obs):
        scv = scaux.get_random_idle_worker(obs, sc2_env.Race.terran)
        mineral = random.choice(
            scaux.get_neutral_units_by_type(obs, units.Neutral.MineralField))
        if scv != scaux._NO_UNITS:
            self.pending_actions.append(
                sc2_actions["Harvest_Gather_unit"].run('queued', scv.tag, mineral.tag))

    def select_random_scv(self, obs):
        # get SCV list
        scvs = scaux.get_units_by_type(obs, units.Terran.SCV)
        length = len(scvs)
        scv = scvs[random.randint(0, length - 1)]
        return scv

    def build_supply_depot_(self, obs):
        random_coord = random.choice(self.supply_depot_coords)
        x, y = random_coord['x'], random_coord['y']
        scv = self.select_random_scv(obs)
        # append action to build supply depot
        self.pending_actions.append(
            sc2_actions["Build_SupplyDepot_pt"].run('now', scv.tag, [x, y]))

    def build_barrack_(self, obs):
        coord = random.choice(self.barrack_coords)
        x, y = coord['x'], coord['y']
        scv = self.select_random_scv(obs)
        # append action to build barrack
        self.pending_actions.append(
            sc2_actions["Build_Barracks_pt"].run('now', scv.tag, [x, y]))


    def build_marine_(self, obs):
        barracks = scaux.get_units_by_type(obs, units.Terran.Barracks)
        if len(barracks) > 0:
            barrack = random.choice(barracks)
            self.pending_actions.append(
                sc2_actions["Train_Marine_quick"].run('now', barrack.tag))