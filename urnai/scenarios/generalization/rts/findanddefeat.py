import math

from urnai.utils.constants import RTSGeneralization

from .collectables import GeneralizedCollectablesScenario


class GeneralizedFindAndDefeatScenario(GeneralizedCollectablesScenario):
    GAME_DEEP_RTS = 'drts'
    GAME_STARCRAFT_II = 'sc2'

    TRAINING_METHOD_SINGLE_ENV = 'single_environment'
    TRAINING_METHOD_MULTIPLE_ENV = 'multiple_environment'

    def __init__(self, game=GAME_DEEP_RTS, render=False,
                 drts_map='total-64x64-playable-43x31-findanddefeat.json',
                 sc2_map='FindAndDefeatZerglings', drts_number_of_players=2, drts_start_oil=99999,
                 drts_start_gold=99999, drts_start_lumber=99999, drts_start_food=99999,
                 fit_to_screen=False, method=TRAINING_METHOD_SINGLE_ENV,
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

        self.drts_attack_radius = 5

    def start(self):
        super().start()

    def step(self, action):
        if (self.game == GeneralizedCollectablesScenario.GAME_DEEP_RTS):
            if self.steps == 0:
                self.setup_map()

            if self.solve_action(action):
                state, reward, done = self.env.step(self.drts_action_noaction)
            else:
                state, reward, done = self.env.step(action)

            if len(self.get_player_units(0)) == 0:
                done = True

            self.steps += 1
            return state, reward, done

        elif (self.game == GeneralizedCollectablesScenario.GAME_STARCRAFT_II):
            self.steps += 1
            return self.env.step(action)

    def setup_map(self):
        tiles = self.env.game.tilemap.tiles
        player = 1
        for i in range(len(tiles)):
            if self.env.game.players[player].num_archer > 23:
                break
            self.random_spawn_archer(player)

    def solve_action(self, action):
        if (action == self.drts_action_attack):
            self.attack_closest_enemy(radius=self.drts_attack_radius)
            return True
        else:
            return super().solve_action(action)

    def get_abs_dist(self, x1, y1, x2, y2):
        return abs(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

    def get_closest_enemy_unit(self, player, radius):
        closest_unit = None
        closest_dist = None
        army_x, army_y = self.get_army_mean(player)

        for enemy in self.env.game.players:
            if enemy.get_id() != player:
                for unit in self.get_player_units(enemy.get_id()):
                    try:
                        unit_x = unit.tile.x
                        unit_y = unit.tile.y
                        dist = self.get_abs_dist(army_x, army_y, unit_x, unit_y)
                        if dist <= radius:
                            if closest_dist is None:
                                closest_dist = dist
                                closest_unit = unit
                            elif dist < closest_dist:
                                closest_dist = dist
                                closest_unit = unit
                    except AttributeError as ae:
                        if "'NoneType' object has no attribute 'x'" not in str(ae):
                            raise

        return closest_unit

    def attack_closest_enemy(self, radius):
        player = 0
        closest_unit = self.get_closest_enemy_unit(player, radius)
        if closest_unit is not None:
            x = closest_unit.tile.x
            y = closest_unit.tile.y
            self.env.game.players[player].right_click(x, y)

    def random_spawn_archer(self, player):
        tile = self.random_tile()

        while not tile.is_buildable:
            tile = self.random_tile()

        if self.env.game.players[player].num_archer < 23:
            self.env.game.players[player].spawn_unit(self.env.constants.Unit.Archer, tile)

    def reset(self):
        self.steps = 0
        self.alternate_envs()
        state = self.env.reset()
        return state
