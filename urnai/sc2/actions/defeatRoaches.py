import math
from statistics import mean
from enum import Enum, auto

from pysc2.env import sc2_env

from urnai.sc2.actions import sc2_actions_aux as scaux
from urnai.sc2.actions.sc2_actions import raw_functions_classes as sc2_actions
from .experiments import ExperimentsActionSpace
from .library_sc2 import move_left, move_right, move_up, move_down

class Actions(Enum):
    MOVELEFT = auto()
    MOVERIGHT = auto()
    MOVEUP = auto()
    MOVEDOWN = auto()
    ATTACK = auto()

class DefeatRoachesActionSpace(ExperimentsActionSpace):

    def __init__(self):
        super().__init__()
        self.hor_threshold = 2
        self.ver_threshold = 2
        self.maximum_attack_range = 16

        self.actions = [member for member in Actions]
        self.named_actions = [member.name for member in Actions]
        self.action_indices = range(len(self.actions))

    def solve_action(self, action_idx, obs):
        if action_idx is not None:
            if action_idx is not self.noaction:
                action = self.actions[action_idx]
                if action == Actions.MOVELEFT:
                    self.do_action(move_left(obs))
                elif action == Actions.MOVERIGHT:
                    self.do_action(move_right(obs))
                elif action == Actions.MOVEUP:
                    self.do_action(move_up(obs))
                elif action == Actions.MOVEDOWN:
                    self.do_action(move_down(obs))
                elif action == Actions.ATTACK:
                    self.do_action(self.attack(obs))
        else:
            self.reset()

    def get_nearest_enemy_unit_inside_radius(self, x, y, obs, radius):
        enemy_army = [unit for unit in obs.raw_units if unit.owner != 1]

        closest_dist = math.inf
        closest_unit = None
        for unit in enemy_army:
            xaux = unit.x
            yaux = unit.y

            dist = abs(math.hypot(x - xaux, y - yaux))

            if dist <= closest_dist and dist <= radius:
                closest_dist = dist
                closest_unit = unit

        if closest_unit is not None:
            return closest_unit

    def get_race_unit_avg(self, obs, race):
        army = scaux.select_army(obs, race)

        xs, ys = [], []
        for unit in army:
            try:
                xs.append(unit.x)
                ys.append(unit.y)
            except AttributeError as ae:
                if "'str' object has no attribute" not in str(ae):
                    raise

        army_x = int(mean(xs))
        army_y = int(mean(ys))
        return army_x, army_y

    def attack_nearest_inside_radius(self, obs, radius):
        race = sc2_env.Race.terran
        army_x, army_y = self.get_race_unit_avg(obs, race)

        nearest_enemy_unit = self.get_nearest_enemy_unit_inside_radius(
            army_x, army_y, obs, radius)

        if nearest_enemy_unit is not None:
            army = scaux.select_army(obs, race)

            return_actions = []
            for unit in army:
                return_actions.append(
                    sc2_actions['Attack_unit'].run('now', unit.tag,
                                            nearest_enemy_unit.tag))
                
            return return_actions

    def attack(self, obs):
        return self.attack_nearest_inside_radius(obs, self.maximum_attack_range)

