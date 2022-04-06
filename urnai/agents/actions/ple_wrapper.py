from .base.abwrapper import ActionWrapper


class PLEWrapper(ActionWrapper):

    def __init__(self, action_set, action_space_size):
        self.move_number = 0
        self.actions = action_set
        self.action_indices = [idx for idx in range(action_space_size)]

    def is_action_done(self):
        return True

    def reset(self):
        self.move_number = 0

    def get_actions(self):
        return self.action_indices

    def get_excluded_actions(self, obs):
        return []

    def get_action(self, action_idx, obs):
        return self.actions[action_idx]
