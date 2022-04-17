from urnai.utils.error import ActionError

from .base.abwrapper import ActionWrapper


class GymWrapper(ActionWrapper):

    def __init__(self, gym_env_actions):
        if gym_env_actions is not None:
            self.move_number = 0
            self.actions = [action_idx for action_idx in range(gym_env_actions)]
        else:
            raise ActionError('Action Space Size must not be None!')

    def is_action_done(self):
        return True

    def reset(self):
        self.move_number = 0

    def get_actions(self):
        return self.actions

    def get_excluded_actions(self, obs):
        return []

    def get_action(self, action_idx, obs):
        return self.actions[action_idx]
