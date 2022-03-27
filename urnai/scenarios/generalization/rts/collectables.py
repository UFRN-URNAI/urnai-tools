import random
from statistics import mean
import sys

from absl import flags
import numpy as np
from pysc2.env import sc2_env
from urnai.agents.actions.scenarios.rts.generalization.all_scenarios import \
    MultipleScenarioActionWrapper
from urnai.agents.rewards.scenarios.rts.generalization.all_scenarios import \
    MultipleScenarioRewardBuilder
from urnai.agents.states.scenarios.rts.generalization.all_scenarios import \
    MultipleScenarioStateBuilder
from urnai.envs.deep_rts import DeepRTSEnv
from urnai.envs.sc2 import SC2Env
from urnai.scenarios.base.abscenario import ABScenario
from urnai.utils.constants import RTSGeneralization
from urnai.utils.error import EnvironmentNotSupportedError, UnsupportedTrainingMethodError


class GeneralizedCollectablesScenario(ABScenario):
    GAME_DEEP_RTS = 'drts'
    GAME_STARCRAFT_II = 'sc2'

    TRAINING_METHOD_SINGLE_ENV = 'single'
    TRAINING_METHOD_MULTIPLE_ENV = 'multiple'

    def __init__(self, game=GAME_DEEP_RTS, render=False,
                 drts_map='total-64x64-playable-16x22-collectables.json',
                 sc2_map='CollectMineralShards', drts_number_of_players=1, drts_start_oil=99999,
                 drts_start_gold=99999, drts_start_lumber=99999, drts_start_food=99999,
                 fit_to_screen=False, method=TRAINING_METHOD_SINGLE_ENV,
                 state_builder_method=RTSGeneralization.STATE_MAP, updates_per_action=12,
                 step_mul=32,
                 map_reduction_factor=RTSGeneralization.STATE_MAP_DEFAULT_REDUCTIONFACTOR,
                 trim_map=False, sc2_real_time_rendering=False):
        self.state_builder_method = state_builder_method
        self.game = game
        self.steps = 0
        self.drts_hor_threshold = 2
        self.drts_ver_threshold = 2
        self.drts_action_previousunit = 0
        self.drts_action_nextunit = 1
        self.drts_action_moveleft = 2
        self.drts_action_moveright = 3
        self.drts_action_moveup = 4
        self.drts_action_movedown = 5
        self.drts_action_moveupleft = 6
        self.drts_action_moveupright = 7
        self.drts_action_movedownleft = 8
        self.drts_action_movedownright = 9
        self.drts_action_attack = 10
        self.drts_action_harvest = 11
        self.drts_action_build0 = 12
        self.drts_action_build1 = 13
        self.drts_action_build2 = 14
        self.drts_action_noaction = 15
        self.sc2_real_time_rendering = sc2_real_time_rendering

        self.map_reduction_factor = map_reduction_factor
        self.trim_map = trim_map

        self.envs = None

        self.method = method
        if method == GeneralizedCollectablesScenario.TRAINING_METHOD_SINGLE_ENV:
            if game == GeneralizedCollectablesScenario.GAME_DEEP_RTS:
                self.env = self.get_drts_env(render, drts_map, drts_number_of_players,
                                             drts_start_oil, drts_start_gold, drts_start_lumber,
                                             drts_start_food,
                                             fit_to_screen, updates_per_action)
            elif game == GeneralizedCollectablesScenario.GAME_STARCRAFT_II:
                self.env = self.get_starcraft_env(sc2_map, render, step_mul)
            else:
                err = """{} only supports the following environments:
        GeneralizedCollectablesScenario.GAME_DEEP_RTS
        GeneralizedCollectablesScenario.GAME_STARCRAFT_II""".format(self.__class__.__name__)
                raise EnvironmentNotSupportedError(err)
        elif method == GeneralizedCollectablesScenario.TRAINING_METHOD_MULTIPLE_ENV:
            env_drts = self.get_drts_env(render, drts_map, drts_number_of_players,
                                         drts_start_oil, drts_start_gold, drts_start_lumber,
                                         drts_start_food,
                                         fit_to_screen, updates_per_action)
            env_sc2 = self.get_starcraft_env(sc2_map, render, step_mul)
            self.envs = {GeneralizedCollectablesScenario.GAME_DEEP_RTS: env_drts,
                         GeneralizedCollectablesScenario.GAME_STARCRAFT_II: env_sc2}
            # games are inverted here because of the way
            # that trainer works
            # after assigning the scenario to be the env
            # trainer calls the reset() method which swap environments
            # so, if what the user wants is the first game to be
            # deeprts, the chosen game here is starcraft
            # when trainer calls the reset() method, then
            # the environment is replaced with the correct game
            # this is of course not the most elegant way
            # to do this, but it works for now
            if game == GeneralizedCollectablesScenario.GAME_DEEP_RTS:
                self.env = self.envs[GeneralizedCollectablesScenario.GAME_STARCRAFT_II]
                self.game = GeneralizedCollectablesScenario.GAME_STARCRAFT_II
            elif game == GeneralizedCollectablesScenario.GAME_STARCRAFT_II:
                self.env = self.envs[GeneralizedCollectablesScenario.GAME_DEEP_RTS]
                self.game = GeneralizedCollectablesScenario.GAME_DEEP_RTS
            else:
                err = """{} only supports the following environments:
        GeneralizedCollectablesScenario.GAME_DEEP_RTS
        GeneralizedCollectablesScenario.GAME_STARCRAFT_II""".format(self.__class__.__name__)
                raise EnvironmentNotSupportedError(err)
        else:
            raise UnsupportedTrainingMethodError(
                "You should use only '{metA}' or '{metB}' as training method.".format(
                    metA=GeneralizedCollectablesScenario.TRAINING_METHOD_SINGLE_ENV,
                    metB=GeneralizedCollectablesScenario.TRAINING_METHOD_MULTIPLE_ENV,
                ))

        self.start()
        self.steps = 0
        self.pickle_black_list = ['game']

    def get_drts_env(self, render, drts_map, drts_number_of_players,
                     drts_start_oil, drts_start_gold, drts_start_lumber, drts_start_food,
                     fit_to_screen, updates_per_action):
        env = DeepRTSEnv(render=render, map=drts_map, updates_per_action=updates_per_action,
                         number_of_players=drts_number_of_players, start_oil=drts_start_oil,
                         start_gold=drts_start_gold, start_lumber=drts_start_lumber,
                         start_food=drts_start_food, fit_to_screen=fit_to_screen,
                         flatten_state=True)
        return env

    def get_starcraft_env(self, sc2_map, render, step_m):
        FLAGS = flags.FLAGS
        FLAGS(sys.argv)
        players = [sc2_env.Agent(sc2_env.Race.terran)]
        env = SC2Env(map_name=sc2_map, render=render, players=players, step_mul=step_m,
                     realtime=self.sc2_real_time_rendering)
        return env

    def get_army_mean(self, player):
        player_units = self.get_player_units(player)
        xs = []
        ys = []

        for unit in player_units:
            try:
                if unit.id != 1 and unit.id != 2:
                    xs.append(unit.tile.x)
                    ys.append(unit.tile.y)
            except AttributeError as ae:
                if "'NoneType' object has no attribute 'x'" not in str(ae):
                    raise

        if len(xs) > 0 and len(ys) > 0:
            army_x = int(mean(xs))
            army_y = int(mean(ys))
            return army_x, army_y
        else:
            return 0, 0

    def solve_action(self, action):
        player = 0
        if action == self.drts_action_moveup:
            self.move_troops_up(player)
            return True
        elif action == self.drts_action_movedown:
            self.move_troops_down(player)
            return True
        elif action == self.drts_action_moveleft:
            self.move_troops_left(player)
            return True
        elif action == self.drts_action_moveright:
            self.move_troops_right(player)
            return True
        return False

    def move_troops_up(self, player):
        cur_x, cur_y = self.get_army_mean(player)

        new_x = cur_x
        new_y = cur_y - self.drts_ver_threshold

        self.move_troops(new_x, new_y)
        # self.env.game.players[player].right_click(new_x, new_y)

    def move_troops_down(self, player):
        cur_x, cur_y = self.get_army_mean(player)

        new_x = cur_x
        new_y = cur_y + self.drts_ver_threshold

        self.move_troops(new_x, new_y)

    def move_troops_left(self, player):
        cur_x, cur_y = self.get_army_mean(player)

        new_x = cur_x - self.drts_hor_threshold
        new_y = cur_y

        self.move_troops(new_x, new_y)

    def move_troops_right(self, player):
        cur_x, cur_y = self.get_army_mean(player)

        new_x = cur_x + self.drts_hor_threshold
        new_y = cur_y

        self.move_troops(new_x, new_y)

    def move_troops(self, new_x, new_y):
        for unit in self.get_player_units(0):
            if unit.tile is not None and new_x > 0 and new_y > 0:
                tile = self.env.game.tilemap.get_tile(new_x, new_y)
                unit.right_click(tile)

    def setup_map(self):
        if self.game == GeneralizedCollectablesScenario.GAME_DEEP_RTS:
            self.collectables_map = self.set_collectable_map()

    def get_player_units(self, player):
        player_units = []
        for unit in self.env.game.units:
            if unit.get_player().get_id() == player:
                player_units.append(unit)

        return player_units

    def get_player_specific_type_units(self, player, unit_id):
        all_units = self.get_player_units(player)
        specific_units = []

        for unit in all_units:
            if unit.type == int(unit_id):
                specific_units.append(unit)

        return specific_units

    def get_tiles_by_id(self, tile_id):
        tiles = []

        for tile in self.env.game.tilemap.tiles:
            if tile.get_type_id() == tile_id:
                tiles.append(tile)

        return tiles

    def alternate_envs(self):
        if self.envs is not None \
                and self.method == GeneralizedCollectablesScenario.TRAINING_METHOD_MULTIPLE_ENV:
            if self.env == self.envs[GeneralizedCollectablesScenario.GAME_DEEP_RTS]:
                self.env = self.envs[GeneralizedCollectablesScenario.GAME_STARCRAFT_II]
                self.game = GeneralizedCollectablesScenario.GAME_STARCRAFT_II
            else:
                self.env = self.envs[GeneralizedCollectablesScenario.GAME_DEEP_RTS]
                self.game = GeneralizedCollectablesScenario.GAME_DEEP_RTS

    def start(self):
        self.env.start()
        self.done = self.env.done

    def step(self, action):
        if self.game == GeneralizedCollectablesScenario.GAME_DEEP_RTS:
            if self.steps == 0:
                self.setup_map()

            if not self.solve_action(action):
                state, reward, done = self.env.step(self.drts_action_noaction)
            else:
                state, reward, done = self.env.step(action)

            player = 0

            reward = 0
            tile_value = 0
            for unit in self.get_player_units(player):
                unit_x = unit.tile.x
                unit_y = unit.tile.y

                tile_value += self.collectables_map[unit_y, unit_x]
                if (tile_value > 0):
                    reward += tile_value
                    self.collectables_map[unit_y, unit_x] = 0

            self.steps += 1
            state['collectables_map'] = self.collectables_map

            return state, reward, done
        elif self.game == GeneralizedCollectablesScenario.GAME_STARCRAFT_II:
            self.steps += 1
            return self.env.step(action)

    def close(self):
        self.env.close()
        self.done = self.env.done

    def reset(self):
        self.steps = 0
        self.alternate_envs()
        state = self.env.reset()
        state['collectables_map'] = self.set_collectable_map()
        return state

    def restart(self):
        self.reset()

    def get_default_reward_builder(self):
        wrapper = MultipleScenarioRewardBuilder(self.__class__.__name__)
        return wrapper

    def get_default_action_wrapper(self):
        wrapper = MultipleScenarioActionWrapper(self.__class__.__name__, self.game, self.method)
        return wrapper

    def get_default_state_builder(self):
        wrapper = MultipleScenarioStateBuilder(self.__class__.__name__,
                                               method=self.state_builder_method)
        wrapper.state_builder.map_reduction_factor = self.map_reduction_factor
        wrapper.trim_map = self.trim_map
        return wrapper

    def random_tile(self):
        tiles_len = len(self.env.game.tilemap.tiles)
        return self.env.game.tilemap.tiles[random.randint(0, tiles_len - 1)]

    def set_collectable_map(self):
        if self.game == GeneralizedCollectablesScenario.GAME_DEEP_RTS:
            width = self.env.game.map.map_width
            height = self.env.game.map.map_height
            map_ = np.zeros((width, height))

            while np.sum(map_) < RTSGeneralization.STATE_MAXIMUM_NUMBER_OF_MINERAL_SHARDS:
                tile = self.random_tile()

                while not tile.is_walkable() or map_[tile.y, tile.x] != 0:
                    tile = self.random_tile()

                map_[tile.y, tile.x] = 1

            return map_
