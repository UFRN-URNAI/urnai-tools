import random

from pysc2.env import sc2_env
from pysc2.lib import actions, units
from urnai.agents.actions import sc2 as scaux
from urnai.agents.rewards.scenarios.rts.generalization.buildunits import \
    BuildUnitsGeneralizedRewardBuilder

from .defeatenemies import DefeatEnemiesDeepRTSActionWrapper, DefeatEnemiesStarcraftIIActionWrapper


class BuildUnitsDeepRTSActionWrapper(DefeatEnemiesDeepRTSActionWrapper):
    def __init__(self):
        super().__init__()
        self.do_nothing = 17
        self.build_farm = 18
        self.build_barrack = 19
        self.build_footman = 20

        self.final_actions = [self.do_nothing, self.build_farm, self.build_barrack,
                              self.build_footman]
        self.named_actions = ['do_nothing', 'build_farm', 'build_barrack', 'build_footman']
        self.action_indices = range(len(self.final_actions))

    def solve_action(self, action_idx, obs):
        if action_idx is not None:
            if action_idx != self.noaction:
                i = action_idx
                self.action_queue.append(self.final_actions[i])
        else:
            # if action_idx was None, this means that the actionwrapper
            # was not resetted properly, so I will reset it here
            # this is not the best way to fix this
            # but until we cannot find why the agent is
            # not resetting the action wrapper properly
            # i'm gonna leave this here
            self.reset()


class BuildUnitsStarcraftIIActionWrapper(DefeatEnemiesStarcraftIIActionWrapper):
    SUPPLY_DEPOT_X = 42
    SUPPLY_DEPOT_Y = 42
    BARRACK_X = 39
    BARRACK_Y = 36

    ACTION_DO_NOTHING = 7
    ACTION_BUILD_SUPPLY_DEPOT = 8
    ACTION_BUILD_BARRACK = 9
    ACTION_BUILD_MARINE = 10

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

        self.do_nothing = BuildUnitsStarcraftIIActionWrapper.ACTION_DO_NOTHING
        self.build_supply_depot = BuildUnitsStarcraftIIActionWrapper.ACTION_BUILD_SUPPLY_DEPOT
        self.build_barrack = BuildUnitsStarcraftIIActionWrapper.ACTION_BUILD_BARRACK
        self.build_marine = BuildUnitsStarcraftIIActionWrapper.ACTION_BUILD_MARINE
        self.actions = [self.do_nothing, self.build_supply_depot, self.build_barrack,
                        self.build_marine]
        self.named_actions = ['do_nothing', 'build_supply_depot', 'build_barrack', 'build_marine']
        self.action_indices = range(len(self.actions))
        self.barrack_coords = BuildUnitsStarcraftIIActionWrapper.MAP_PLAYER_BARRACK_COORDINATES
        self.supply_depot_coords = \
            BuildUnitsStarcraftIIActionWrapper.MAP_PLAYER_SUPPLY_DEPOT_COORDINATES

    def solve_action(self, action_idx, obs):
        if action_idx is not None:
            if action_idx != self.noaction:
                BuildUnitsGeneralizedRewardBuilder.LAST_CHOSEN_ACTION = self.actions[action_idx]
                action = self.actions[action_idx]
                if action == self.do_nothing:
                    self.collect_idle(obs)
                elif action == self.build_supply_depot:
                    self.build_supply_depot_(obs)
                elif action == self.build_barrack:
                    self.build_barrack_(obs)
                elif action == self.build_marine:
                    self.build_marine_(obs)
                elif action == self.stop:
                    self.pending_actions.clear()
        else:
            # if action_idx was None, this means that the actionwrapper
            # was not resetted properly, so I will reset it here
            # this is not the best way to fix this
            # but until we cannot find why the agent is
            # not resetting the action wrapper properly
            # i'm gonna leave this here
            self.reset()

    def collect(self, obs):
        # get SCV list
        scvs = scaux.get_units_by_type(obs, units.Terran.SCV)
        # get mineral list
        mineral_fields = scaux.get_neutral_units_by_type(obs, units.Neutral.MineralField)
        # split SCVs into sets of numberSCVs/numberOfMinerals
        n = int(len(scvs) / len(mineral_fields))
        scvs_sets = [scvs[i * n:(i + 1) * n] for i in range((len(scvs) + n - 1) // n)]
        # make every set of SCVs collect one mineral
        for i in range(len(mineral_fields)):
            mineral = mineral_fields[i]
            scvset = scvs_sets[i]
            for scv in scvset:
                self.pending_actions.append(
                    actions.RAW_FUNCTIONS.Harvest_Gather_unit('queued', scv.tag, mineral.tag))

    def collect_idle(self, obs):
        scv = scaux.get_random_idle_worker(obs, sc2_env.Race.terran)
        mineral = random.choice(scaux.get_neutral_units_by_type(obs, units.Neutral.MineralField))
        if scv != scaux._NO_UNITS:
            self.pending_actions.append(
                actions.RAW_FUNCTIONS.Harvest_Gather_unit('queued', scv.tag, mineral.tag))

    def select_random_scv(self, obs):
        # get SCV list
        scvs = scaux.get_units_by_type(obs, units.Terran.SCV)
        length = len(scvs)
        scv = scvs[random.randint(0, length - 1)]
        return scv

    def build_supply_depot_(self, obs):
        coord = random.choice(self.supply_depot_coords)
        x, y = coord['x'], coord['y']
        scv = self.select_random_scv(obs)
        # append action to build supply depot
        self.pending_actions.append(
            actions.RAW_FUNCTIONS.Build_SupplyDepot_pt('now', scv.tag, [x, y]))

    def build_barrack_(self, obs):
        coord = random.choice(self.barrack_coords)
        x, y = coord['x'], coord['y']
        scv = self.select_random_scv(obs)
        # append action to build supply depot
        self.pending_actions.append(actions.RAW_FUNCTIONS.Build_Barracks_pt('now', scv.tag, [x, y]))

    def build_marine_(self, obs):
        barracks = scaux.get_units_by_type(obs, units.Terran.Barracks)
        if len(barracks) > 0:
            barrack = random.choice(barracks)
            self.pending_actions.append(
                actions.RAW_FUNCTIONS.Train_Marine_quick('now', barrack.tag))
