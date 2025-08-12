from pysc2.env import sc2_env

from urnai.actions.action_space_base import ActionSpaceBase
from urnai.sc2.actions import sc2_actions_aux as scaux
from urnai.sc2.actions.sc2_actions import raw_functions_classes as sc2_actions


class CollectablesActionSpace(ActionSpaceBase):

    def __init__(self):
        self.noaction = [sc2_actions["no_op"].run()]

        self.excluded_actions = []

        min_x, max_x = 22, 43
        min_y, max_y = 28, 43
        num_x = max_x - min_x + 1  # 22
        num_y = max_y - min_y + 1  # 16
        total_actions = num_x * num_y

        self.actions = list(range(total_actions))
        self.named_actions = [
            f"move_to_{min_x + (i % num_x)}_{min_y + (i // num_x)}"
            for i in range(total_actions)
        ]
        self.action_indices = list(range(total_actions))

        self.pending_actions = []

    def is_action_done(self):
        return len(self.pending_actions) == 0

    def reset(self):
        self.pending_actions = []

    def get_actions(self):
        return self.action_indices

    def get_excluded_actions(self, obs):
        return self.excluded_actions

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
                army_x, army_y = self.calc_action_coordinates(action_idx)
                self.move_x_y(obs, army_x, army_y)
        else:
            self.reset()
    
    def calc_action_coordinates(self, action_idx):
        min_x, max_x = 22, 43
        min_y, max_y = 28, 43
        num_x = max_x - min_x + 1  # 22
        num_y = max_y - min_y + 1  # 16

        if not (0 <= action_idx < num_x * num_y):
            raise ValueError(f"action_idx {action_idx} out of range "+
                             f"(0 to {num_x * num_y - 1})")

        x = min_x + (action_idx % num_x)
        y = min_y + (action_idx // num_x)
        return x, y

    def move_x_y(self, obs, x, y):
        army = scaux.select_army(obs, sc2_env.Race.terran)

        for unit in army:
            self.pending_actions.append(
                sc2_actions["Move_pt"].run(
                    'now', unit.tag,[x, y]))

    def get_named_actions(self):
        return self.named_actions