import random
from sys import maxsize as maxint

from urnai.utils.constants import RTSGeneralization

from .findanddefeat import GeneralizedFindAndDefeatScenario


class GeneralizedDefeatEnemiesScenario(GeneralizedFindAndDefeatScenario):
    GAME_DEEP_RTS = 'drts'
    GAME_STARCRAFT_II = 'sc2'
    MAP_1 = 'map1'
    MAP_2 = 'map2'

    MAP_1_PLAYER_X = 39
    MAP_2_PLAYER_X = 26
    Y_PLAYER_BASE = 30
    MAP_PLAYER_LOCATIONS = {
        MAP_1: [
            {'x': MAP_1_PLAYER_X, 'y': Y_PLAYER_BASE},
            {'x': MAP_1_PLAYER_X, 'y': Y_PLAYER_BASE + 1},
            {'x': MAP_1_PLAYER_X, 'y': Y_PLAYER_BASE + 2},
            {'x': MAP_1_PLAYER_X, 'y': Y_PLAYER_BASE + 3},
            {'x': MAP_1_PLAYER_X, 'y': Y_PLAYER_BASE + 4},
            {'x': MAP_1_PLAYER_X, 'y': Y_PLAYER_BASE + 5},
            {'x': MAP_1_PLAYER_X, 'y': Y_PLAYER_BASE + 6},
            {'x': MAP_1_PLAYER_X, 'y': Y_PLAYER_BASE + 7},
            {'x': MAP_1_PLAYER_X, 'y': Y_PLAYER_BASE + 8},
        ],
        MAP_2: [
            {'x': MAP_2_PLAYER_X, 'y': Y_PLAYER_BASE},
            {'x': MAP_2_PLAYER_X, 'y': Y_PLAYER_BASE + 1},
            {'x': MAP_2_PLAYER_X, 'y': Y_PLAYER_BASE + 2},
            {'x': MAP_2_PLAYER_X, 'y': Y_PLAYER_BASE + 3},
            {'x': MAP_2_PLAYER_X, 'y': Y_PLAYER_BASE + 4},
            {'x': MAP_2_PLAYER_X, 'y': Y_PLAYER_BASE + 5},
            {'x': MAP_2_PLAYER_X, 'y': Y_PLAYER_BASE + 6},
            {'x': MAP_2_PLAYER_X, 'y': Y_PLAYER_BASE + 7},
            {'x': MAP_2_PLAYER_X, 'y': Y_PLAYER_BASE + 8},
        ],
    }

    MAP_1_ENEMY_X = 24
    MAP_2_ENEMY_X = 42
    Y_ENEMY_BASE = 36
    MAP_ENEMY_LOCATIONS = {
        MAP_1: [
            {'x': MAP_1_ENEMY_X, 'y': Y_ENEMY_BASE},
            {'x': MAP_1_ENEMY_X, 'y': Y_ENEMY_BASE + 1},
            {'x': MAP_1_ENEMY_X, 'y': Y_ENEMY_BASE + 2},
            {'x': MAP_1_ENEMY_X, 'y': Y_ENEMY_BASE + 3},
        ],
        MAP_2: [
            {'x': MAP_2_ENEMY_X, 'y': Y_ENEMY_BASE},
            {'x': MAP_2_ENEMY_X, 'y': Y_ENEMY_BASE + 1},
            {'x': MAP_2_ENEMY_X, 'y': Y_ENEMY_BASE + 2},
            {'x': MAP_2_ENEMY_X, 'y': Y_ENEMY_BASE + 3},
        ],
    }

    MAP_X_CORNER_LEFT = 21
    MAP_X_CORNER_RIGHT = 44
    MAP_Y_CORNER_UP = 27
    MAP_Y_CORNER_DOWN = 50

    TRAINING_METHOD_SINGLE_ENV = 'single_environment'
    TRAINING_METHOD_MULTIPLE_ENV = 'multiple_environment'

    def __init__(self, game=GAME_DEEP_RTS, render=False,
                 drts_map='total-64x64-playable-22x16-defeatenemies.json', sc2_map='DefeatRoaches',
                 drts_number_of_players=2, drts_start_oil=999999, drts_start_gold=999999,
                 drts_start_lumber=999999, drts_start_food=999999, fit_to_screen=False,
                 method=TRAINING_METHOD_SINGLE_ENV,
                 state_builder_method=RTSGeneralization.STATE_MAP, updates_per_action=12,
                 step_mul=8, sc2_real_time_rendering=False):
        super().__init__(game=game, render=render, drts_map=drts_map, sc2_map=sc2_map,
                         drts_number_of_players=drts_number_of_players,
                         drts_start_oil=drts_start_oil, drts_start_gold=drts_start_gold,
                         drts_start_lumber=drts_start_lumber, drts_start_food=drts_start_food,
                         fit_to_screen=fit_to_screen, method=method,
                         state_builder_method=state_builder_method,
                         updates_per_action=updates_per_action, step_mul=step_mul,
                         sc2_real_time_rendering=sc2_real_time_rendering)
        self.drts_attack_radius = maxint
        self.drts_hor_threshold = 1
        self.drts_ver_threshold = 1
        self.drts_action_run = 17

    def setup_map(self):
        if self.game == GeneralizedDefeatEnemiesScenario.GAME_DEEP_RTS:
            choice = random.randint(0, 1)

            if choice == 0:
                self.map_spawn = GeneralizedDefeatEnemiesScenario.MAP_1
            else:
                self.map_spawn = GeneralizedDefeatEnemiesScenario.MAP_2

            self.spawn_army()

    def solve_action(self, action):
        if (action == self.drts_action_run):
            self.run_away()
            return True
        else:
            return super().solve_action(action)

    def run_away(self):
        player = 0
        enemy = 1
        p_army_x, p_army_y = self.get_army_mean(player)
        e_army_x, e_army_y = self.get_army_mean(enemy)

        new_x = 0
        new_y = 0

        if p_army_x - e_army_x < 0:
            new_x = GeneralizedDefeatEnemiesScenario.MAP_X_CORNER_LEFT
        else:
            new_x = GeneralizedDefeatEnemiesScenario.MAP_X_CORNER_RIGHT

        if p_army_y - e_army_y < 0:
            new_y = GeneralizedDefeatEnemiesScenario.MAP_Y_CORNER_UP
        else:
            new_y = GeneralizedDefeatEnemiesScenario.MAP_Y_CORNER_DOWN

        self.env.game.players[player].right_click(new_x, new_y)

    def spawn_army(self):
        for coords in GeneralizedDefeatEnemiesScenario.MAP_PLAYER_LOCATIONS[self.map_spawn]:
            tile = self.env.game.tilemap.get_tile(coords['x'], coords['y'])
            self.env.game.players[0].spawn_unit(self.env.constants.Unit.Archer, tile)

        for coords in GeneralizedDefeatEnemiesScenario.MAP_ENEMY_LOCATIONS[self.map_spawn]:
            tile = self.env.game.tilemap.get_tile(coords['x'], coords['y'])
            self.env.game.players[1].spawn_unit(self.env.constants.Unit.Archer, tile)
