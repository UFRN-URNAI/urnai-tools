from statistics import mean

from pysc2.env import sc2_env

from urnai.actions.action_space_base import ActionSpaceBase
from urnai.sc2.actions import sc2_actions_aux as scaux
from urnai.sc2.actions.sc2_actions import raw_functions_classes as sc2_actions


class CollectablesActionSpace(ActionSpaceBase):

    def __init__(self):
        self.noaction = [sc2_actions["no_op"].run()]
        self.move_number = 0

        self.hor_threshold = 2
        self.ver_threshold = 2

        self.moveleft = 0
        self.moveright = 1
        self.moveup = 2
        self.movedown = 3

        self.excluded_actions = []

        self.actions = [self.moveleft, self.moveright, self.moveup, self.movedown]
        self.named_actions = ['move_left', 'move_right', 'move_up', 'move_down']
        self.action_indices = range(len(self.actions))

        self.pending_actions = []

    def is_action_done(self):
        return len(self.pending_actions) == 0

    def reset(self):
        self.move_number = 0
        self.pending_actions = []

    def get_actions(self):
        return self.action_indices

    def get_excluded_actions(self, obs):
        """Get the excluded actions based on the current observation."""
        excluded = []

        for action in self.actions:
            if not self.is_move_valid(obs, action):
                excluded.append(action)

        return excluded

    def get_action(self, action_idx, obs):
        action = None
        if len(self.pending_actions) == 0:
            action = self.noaction
        else:
            action = [self.pending_actions.pop()]
        self.solve_action(action_idx, obs)
        return action

    def solve_action(self, action_idx, obs):
        if action_idx is not None:
            if action_idx is not self.noaction:
                action = self.actions[action_idx]
                if action == self.moveleft:
                    self.move_left(obs)
                elif action == self.moveright:
                    self.move_right(obs)
                elif action == self.moveup:
                    self.move_up(obs)
                elif action == self.movedown:
                    self.move_down(obs)
        else:
            self.reset()

    def move_left(self, obs):
        army = scaux.select_army(obs, sc2_env.Race.terran)
        xs = [unit.x for unit in army]
        ys = [unit.y for unit in army]

        new_army_x = int(mean(xs)) - self.hor_threshold
        new_army_y = int(mean(ys))

        for unit in army:
            self.pending_actions.append(
                sc2_actions["Move_pt"].run(
                    'now', unit.tag,[new_army_x, new_army_y]))

    def move_right(self, obs):
        army = scaux.select_army(obs, sc2_env.Race.terran)
        xs = [unit.x for unit in army]
        ys = [unit.y for unit in army]

        new_army_x = int(mean(xs)) + self.hor_threshold
        new_army_y = int(mean(ys))

        for unit in army:
            self.pending_actions.append(
                sc2_actions["Move_pt"].run(
                    'now', unit.tag,[new_army_x, new_army_y]))

    def move_down(self, obs):
        army = scaux.select_army(obs, sc2_env.Race.terran)
        xs = [unit.x for unit in army]
        ys = [unit.y for unit in army]

        new_army_x = int(mean(xs))
        new_army_y = int(mean(ys)) + self.ver_threshold

        for unit in army:
            self.pending_actions.append(
                sc2_actions["Move_pt"].run(
                    'now', unit.tag,[new_army_x, new_army_y]))

    def move_up(self, obs):
        army = scaux.select_army(obs, sc2_env.Race.terran)
        xs = [unit.x for unit in army]
        ys = [unit.y for unit in army]

        new_army_x = int(mean(xs))
        new_army_y = int(mean(ys)) - self.ver_threshold

        for unit in army:
            self.pending_actions.append(
                sc2_actions["Move_pt"].run(
                    'now', unit.tag,[new_army_x, new_army_y]))

    def get_action_name_str_by_int(self, action_int):
        action_str = ''
        for attrstr in dir(self):
            attr = getattr(self, attrstr)
            if action_int == attr:
                action_str = attrstr

        return action_str

    def get_no_action(self):
        return self.noaction

    def get_named_actions(self):
        return self.named_actions

    def is_move_valid(self, obs, action):

        # Move left -> Army min positions: X: [22, 23]
        # Move right -> Army max positions: X: [42, 43]
        # Move up -> Army min positions: Y: [28, 29]
        # Move down -> Army max positions: Y: [42, 43]

        army = scaux.select_army(obs, sc2_env.Race.terran)
        xs = [unit.x for unit in army]
        ys = [unit.y for unit in army]

        if (action == self.moveleft and all(x in [22, 23] for x in xs))\
        or (action == self.moveright and all(x in [42, 43] for x in xs))\
        or (action == self.moveup and all(y in [28, 29] for y in ys))\
        or (action == self.movedown and all(y in [42, 43] for y in ys)):
            return False
        return True