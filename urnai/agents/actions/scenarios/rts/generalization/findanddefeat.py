import math
from statistics import mean

from pysc2.env import sc2_env
from pysc2.lib import actions
from urnai.agents.actions import sc2 as scaux

from .collectables import CollectablesDeepRTSActionWrapper, CollectablesStarcraftIIActionWrapper


class FindAndDefeatDeepRTSActionWrapper(CollectablesDeepRTSActionWrapper):
    def __init__(self):
        super().__init__()
        self.cancel = 16

        self.named_actions = None
        self.final_actions = [self.moveleft, self.moveright, self.moveup, self.movedown,
                              self.attack]
        self.action_indices = range(len(self.final_actions))

    def solve_action(self, action_idx, obs):
        # print('>>>>>>>>> AVAIL. ACTIONS: {}'.format(self.final_actions))
        # print('>>>>>>>>> CHOSEN ACTION: {}'.format(self.final_actions[action_idx]))
        if action_idx is not None:
            if action_idx != self.noaction:
                i = action_idx
                if self.final_actions[i] == self.attack:
                    self.attack_(obs)
                elif self.final_actions[i] == self.cancel:
                    self.action_queue.clear()
                else:
                    super().solve_action(action_idx, obs)
        else:
            # if action_idx was None, this means that the actionwrapper
            # was not resetted properly, so I will reset it here
            # this is not the best way to fix this
            # but until we cannot find why the agent is
            # not resetting the action wrapper properly
            # i'm gonna leave this here
            self.reset()

    def attack_(self, obs):
        self.enqueue_action_for_player_units(obs, self.attack)


class FindAndDefeatStarcraftIIActionWrapper(CollectablesStarcraftIIActionWrapper):
    def __init__(self):
        super().__init__()
        self.maximum_attack_range = 2
        self.attack = 4
        self.stop = 6
        self.actions = [self.moveleft, self.moveright, self.moveup, self.movedown, self.attack,
                        self.stop]
        self.action_indices = range(len(self.actions))

    def solve_action(self, action_idx, obs):
        if action_idx is not None:
            if action_idx != self.noaction:
                action = self.actions[action_idx]
                if action == self.moveleft:
                    self.move_left(obs)
                elif action == self.moveright:
                    self.move_right(obs)
                elif action == self.moveup:
                    self.move_up(obs)
                elif action == self.movedown:
                    self.move_down(obs)
                elif action == self.attack:
                    self.attack_(obs)
        else:
            # if action_idx was None, this means that the actionwrapper
            # was not resetted properly, so I will reset it here
            # this is not the best way to fix this
            # but until we cannot find why the agent is
            # not resetting the action wrapper properly
            # i'm gonna leave this here
            self.reset()

    def get_nearest_enemy_unit_inside_radius(self, x, y, obs, radius):
        enemy_army = [unit for unit in obs.raw_units if unit.owner != 1]

        closest_dist = 9999999999999
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
        # get army coordinates
        race = sc2_env.Race.terran
        army_x, army_y = self.get_race_unit_avg(obs, race)

        # get nearest unit
        enemy_unit = self.get_nearest_enemy_unit_inside_radius(army_x, army_y, obs, radius)

        # tell each unit in army to attack nearest enemy
        if enemy_unit is not None:
            army = scaux.select_army(obs, race)
            for unit in army:
                self.pending_actions.append(
                    actions.RAW_FUNCTIONS.Attack_unit('now', unit.tag, enemy_unit.tag))

    def attack_(self, obs):
        self.attack_nearest_inside_radius(obs, self.maximum_attack_range)
