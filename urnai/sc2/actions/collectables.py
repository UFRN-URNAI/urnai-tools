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

        self.move_left = 0
        self.move_right = 1
        self.move_up = 2
        self.move_down = 3

        self.excluded_actions = []

        self.actions = [self.move_left, self.move_right, self.move_up, self.move_down]
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
                if action_idx not in self.actions:
                    raise ValueError(f"Invalid action index: {action_idx}. "+
                                     f"Valid actions: {self.actions}")
                action = self.actions[action_idx]
                if action == self.move_left:
                    self.move_left_(obs)
                elif action == self.move_right:
                    self.move_right_(obs)
                elif action == self.move_up:
                    self.move_up_(obs)
                else:
                    self.move_down_(obs)
        else:
            self.reset()

    def move_left_(self, obs):
        army = scaux.select_army(obs, sc2_env.Race.terran)
        xs = [unit.x for unit in army]
        ys = [unit.y for unit in army]

        new_army_x = int(mean(xs)) - self.hor_threshold
        new_army_y = int(mean(ys))

        for unit in army:
            self.pending_actions.append(
                sc2_actions["Move_pt"].run(
                    'now', unit.tag,[new_army_x, new_army_y]))

    def move_right_(self, obs):
        army = scaux.select_army(obs, sc2_env.Race.terran)
        xs = [unit.x for unit in army]
        ys = [unit.y for unit in army]

        new_army_x = int(mean(xs)) + self.hor_threshold
        new_army_y = int(mean(ys))

        for unit in army:
            self.pending_actions.append(
                sc2_actions["Move_pt"].run(
                    'now', unit.tag,[new_army_x, new_army_y]))

    def move_down_(self, obs):
        army = scaux.select_army(obs, sc2_env.Race.terran)
        xs = [unit.x for unit in army]
        ys = [unit.y for unit in army]

        new_army_x = int(mean(xs))
        new_army_y = int(mean(ys)) + self.ver_threshold

        for unit in army:
            self.pending_actions.append(
                sc2_actions["Move_pt"].run(
                    'now', unit.tag,[new_army_x, new_army_y]))

    def move_up_(self, obs):
        army = scaux.select_army(obs, sc2_env.Race.terran)
        xs = [unit.x for unit in army]
        ys = [unit.y for unit in army]

        new_army_x = int(mean(xs))
        new_army_y = int(mean(ys)) - self.ver_threshold

        for unit in army:
            self.pending_actions.append(
                sc2_actions["Move_pt"].run(
                    'now', unit.tag,[new_army_x, new_army_y]))

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

        if (action == self.move_left and all(x in [22, 23] for x in xs))\
        or (action == self.move_right and all(x in [42, 43] for x in xs))\
        or (action == self.move_up and all(y in [28, 29] for y in ys))\
        or (action == self.move_down and all(y in [42, 43] for y in ys)):
            return False
        return True