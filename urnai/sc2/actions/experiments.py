from abc import abstractmethod

from urnai.actions.action_space_base import ActionSpaceBase
from urnai.sc2.actions.sc2_actions import raw_functions_classes as sc2_actions


class ExperimentsActionSpace(ActionSpaceBase):

    def __init__(self):
        self.noaction = [sc2_actions["no_op"].run()]

        self.excluded_actions = []

        self.actions = []
        self.named_actions = []
        self.action_indices = range(len(self.actions))

        self.pending_actions = []

    def is_action_done(self):
        return len(self.pending_actions) == 0

    def reset(self):
        self.pending_actions = []

    def get_actions(self):
        return self.action_indices

    def get_excluded_actions(self, obs):
        return []

    def get_action(self, action_idx, obs):
        action = None
        if len(self.pending_actions) == 0:
            action = self.noaction
        else:
            action = [self.pending_actions.pop()]
        self.solve_action(action_idx, obs)
        return action

    @abstractmethod
    def solve_action(self, action_idx, obs):
        ...

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

    def do_action(self, action : list):
        if action is not None:
            self.pending_actions += action