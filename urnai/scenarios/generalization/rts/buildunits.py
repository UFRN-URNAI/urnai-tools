import random

from urnai.agents.rewards.scenarios.rts.generalization.buildunits import \
    BuildUnitsGeneralizedRewardBuilder
from urnai.utils.constants import RTSGeneralization
from utils.reporter import Reporter as rp

from .defeatenemies import GeneralizedDefeatEnemiesScenario


class GeneralizedBuildUnitsScenario(GeneralizedDefeatEnemiesScenario):
    GAME_DEEP_RTS = 'drts'
    GAME_STARCRAFT_II = 'sc2'

    TRAINING_METHOD_SINGLE_ENV = 'single_environment'
    TRAINING_METHOD_MULTIPLE_ENV = 'multiple_environment'

    MAP_1 = 'map1'
    MAP_2 = 'map2'

    MAP_PLAYER_TOWNHALL_X = 30
    MAP_PLAYER_TOWNHALL_Y = 36
    MAP_PLAYER_BARRACK_X = 39
    MAP_PLAYER_BARRACK_Y = 36
    MAP_PLAYER_FARM_X = 42
    MAP_PLAYER_FARM_Y = 42
    MAP_PLAYER_FARM_COORDINATES = [
        {'x': MAP_PLAYER_FARM_X, 'y': MAP_PLAYER_FARM_Y},
        {'x': MAP_PLAYER_FARM_X - 1, 'y': MAP_PLAYER_FARM_Y},
        {'x': MAP_PLAYER_FARM_X - 2, 'y': MAP_PLAYER_FARM_Y},
        {'x': MAP_PLAYER_FARM_X - 3, 'y': MAP_PLAYER_FARM_Y},
        {'x': MAP_PLAYER_FARM_X - 4, 'y': MAP_PLAYER_FARM_Y},
        {'x': MAP_PLAYER_FARM_X - 5, 'y': MAP_PLAYER_FARM_Y},
        {'x': MAP_PLAYER_FARM_X - 6, 'y': MAP_PLAYER_FARM_Y},
    ]

    MAP_PLAYER_BARRACK_COORDINATES = [
        {'x': MAP_PLAYER_BARRACK_X, 'y': MAP_PLAYER_BARRACK_Y},
        {'x': MAP_PLAYER_BARRACK_X, 'y': MAP_PLAYER_BARRACK_Y - 6},
    ]

    MAP_1_PLAYER_X = 25
    MAP_2_PLAYER_X = 33
    Y_PLAYER_BASE = 33
    MAP_PLAYER_LOCATIONS = {
        MAP_1: [
            # troops
            {'x': MAP_1_PLAYER_X, 'y': Y_PLAYER_BASE},
            {'x': MAP_1_PLAYER_X, 'y': Y_PLAYER_BASE + 1},
            {'x': MAP_1_PLAYER_X, 'y': Y_PLAYER_BASE + 2},
            {'x': MAP_1_PLAYER_X, 'y': Y_PLAYER_BASE + 3},
            {'x': MAP_1_PLAYER_X, 'y': Y_PLAYER_BASE + 4},
            {'x': MAP_1_PLAYER_X, 'y': Y_PLAYER_BASE + 5},
            {'x': MAP_1_PLAYER_X + 1, 'y': Y_PLAYER_BASE},
            {'x': MAP_1_PLAYER_X + 1, 'y': Y_PLAYER_BASE + 1},
            {'x': MAP_1_PLAYER_X + 1, 'y': Y_PLAYER_BASE + 2},
            {'x': MAP_1_PLAYER_X + 1, 'y': Y_PLAYER_BASE + 3},
            {'x': MAP_1_PLAYER_X + 1, 'y': Y_PLAYER_BASE + 4},
            {'x': MAP_1_PLAYER_X + 1, 'y': Y_PLAYER_BASE + 5},
        ],
        MAP_2: [
            # troops
            {'x': MAP_2_PLAYER_X, 'y': Y_PLAYER_BASE},
            {'x': MAP_2_PLAYER_X, 'y': Y_PLAYER_BASE + 1},
            {'x': MAP_2_PLAYER_X, 'y': Y_PLAYER_BASE + 2},
            {'x': MAP_2_PLAYER_X, 'y': Y_PLAYER_BASE + 3},
            {'x': MAP_2_PLAYER_X, 'y': Y_PLAYER_BASE + 4},
            {'x': MAP_2_PLAYER_X, 'y': Y_PLAYER_BASE + 5},
            {'x': MAP_2_PLAYER_X - 1, 'y': Y_PLAYER_BASE},
            {'x': MAP_2_PLAYER_X - 1, 'y': Y_PLAYER_BASE + 1},
            {'x': MAP_2_PLAYER_X - 1, 'y': Y_PLAYER_BASE + 2},
            {'x': MAP_2_PLAYER_X - 1, 'y': Y_PLAYER_BASE + 3},
            {'x': MAP_2_PLAYER_X - 1, 'y': Y_PLAYER_BASE + 4},
            {'x': MAP_2_PLAYER_X - 1, 'y': Y_PLAYER_BASE + 5},
        ],
    }

    ACTION_DRTS_DO_NOTHING = 17
    ACTION_DRTS_BUILD_FARM = 18
    ACTION_DRTS_BUILD_BARRACK = 19
    ACTION_DRTS_BUILD_FOOTMAN = 20

    def __init__(self, game=GAME_DEEP_RTS, render=False,
                 drts_map='total-64x64-playable-22x16-buildunits.json', sc2_map='BuildMarines',
                 drts_start_oil=999999, drts_start_gold=0, drts_start_lumber=999999,
                 drts_start_food=999999, fit_to_screen=False, method=TRAINING_METHOD_SINGLE_ENV,
                 state_builder_method=RTSGeneralization.STATE_MAP, updates_per_action=6, step_mul=8,
                 sc2_real_time_rendering=False):
        super().__init__(game=game, render=render, drts_map=drts_map, sc2_map=sc2_map,
                         drts_number_of_players=1, drts_start_oil=drts_start_oil,
                         drts_start_gold=drts_start_gold, drts_start_lumber=drts_start_lumber,
                         drts_start_food=drts_start_food, fit_to_screen=fit_to_screen,
                         method=method, state_builder_method=state_builder_method,
                         updates_per_action=updates_per_action, step_mul=step_mul,
                         sc2_real_time_rendering=sc2_real_time_rendering)
        self.farm_spawn_points = GeneralizedBuildUnitsScenario.MAP_PLAYER_FARM_COORDINATES
        self.barrack_spawn_points = GeneralizedBuildUnitsScenario.MAP_PLAYER_BARRACK_COORDINATES

    def step(self, action):
        if (self.game == GeneralizedBuildUnitsScenario.GAME_DEEP_RTS):
            BuildUnitsGeneralizedRewardBuilder.LAST_CHOSEN_ACTION = action
            if self.steps == 0:
                self.setup_map()
                self.spawn_army()
            elif self.steps == 1:
                self.collect_gold()

            if rp.VERBOSITY_LEVEL > 0:
                str_ = """  DRTS Episode Status:
                 Number of gold = {},
                 Number of barracks = {},
                 Number of farms = {},
                 Number of soldiers = {}""".format(
                    self.env.players[0].gold,
                    self.get_drts_unit_type_count(0, self.env.constants.Unit.Barracks),
                    self.get_drts_unit_type_count(0, self.env.constants.Unit.Farm),
                    self.get_drts_unit_type_count(0, self.env.constants.Unit.Footman),
                )
                rp.report(str_, verbosity_lvl=1)
            state, reward, done = None, None, None
            if action == GeneralizedBuildUnitsScenario.ACTION_DRTS_DO_NOTHING:
                no_action = 15
                state, reward, done = self.env.step(no_action)
            elif action == GeneralizedBuildUnitsScenario.ACTION_DRTS_BUILD_FARM:
                self.build_farm()
                no_action = 15
                state, reward, done = self.env.step(no_action)
            elif action == GeneralizedBuildUnitsScenario.ACTION_DRTS_BUILD_BARRACK:
                self.build_barrack()
                no_action = 15
                state, reward, done = self.env.step(no_action)
            elif action == GeneralizedBuildUnitsScenario.ACTION_DRTS_BUILD_FOOTMAN:
                self.build_footman()
                no_action = 15
                state, reward, done = self.env.step(no_action)
            else:
                state, reward, done = self.env.step(action)
            self.steps += 1
            return state, reward, done

        elif (self.game == GeneralizedBuildUnitsScenario.GAME_STARCRAFT_II):
            self.steps += 1
            return self.env.step(action)

    def spawn_army(self):
        for coords in GeneralizedBuildUnitsScenario.MAP_PLAYER_LOCATIONS[self.map_spawn]:
            # idx = GeneralizedBuildUnitsScenario.MAP_PLAYER_LOCATIONS[self.map_spawn].index(coords)
            # ln = len(GeneralizedBuildUnitsScenario.MAP_PLAYER_LOCATIONS[self.map_spawn])
            tile = self.env.game.tilemap.get_tile(coords['x'], coords['y'])
            self.env.game.players[0].spawn_unit(self.env.constants.Unit.Peasant, tile)

        tile = self.env.game.tilemap.get_tile(GeneralizedBuildUnitsScenario.MAP_PLAYER_TOWNHALL_X,
                                              GeneralizedBuildUnitsScenario.MAP_PLAYER_TOWNHALL_Y)
        self.env.game.players[0].spawn_unit(self.env.constants.Unit.TownHall, tile)

    def collect_gold(self):
        player = 0
        peasants = self.get_player_specific_type_units(player, self.env.constants.Unit.Peasant)
        gold = 102
        gold_tiles = self.get_tiles_by_id(gold)
        how_many_gold_spots = len(gold_tiles)
        n = how_many_gold_spots
        peasants_sets = [peasants[i * n:(i + 1) * n] for i in range((len(peasants) + n - 1) // n)]

        for i in range(len(peasants_sets)):
            peasant_set = peasants_sets[i]
            gold_tile = gold_tiles[i]

            for peasant in peasant_set:
                peasant.right_click(gold_tile)

    def get_farm_count(self):
        return self.get_drts_unit_type_count(0, self.env.constants.Unit.Farm)

    def get_drts_unit_type_count(self, player, unit_type):
        player_units = self.get_player_units(player)
        specific_units = []
        for unit in player_units:
            if unit.type == unit_type:
                specific_units.append(unit)

        return len(specific_units)

    def build_farm(self):
        coord = random.choice(self.farm_spawn_points)
        tile = self.env.game.tilemap.get_tile(coord['x'], coord['y'])
        self.env.game.players[0].spawn_unit(self.env.constants.Unit.Farm, tile)

    def build_barrack(self):
        coord = random.choice(self.barrack_spawn_points)
        tile = self.env.game.tilemap.get_tile(coord['x'], coord['y'])
        if self.get_farm_count() > 0:
            self.env.game.players[0].spawn_unit(self.env.constants.Unit.Barracks, tile)

    def build_footman(self):
        player = 0
        barracks_list = self.get_player_specific_type_units(player,
                                                            self.env.constants.Unit.Barracks)
        if len(barracks_list) > 0:
            barracks = random.choice(barracks_list)
            barracks.build(0)

    def reset(self):
        state = super().reset()
        BuildUnitsGeneralizedRewardBuilder.LAST_CHOSEN_ACTION = -1
        return state
