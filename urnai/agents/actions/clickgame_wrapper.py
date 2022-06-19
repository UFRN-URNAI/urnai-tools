from math import floor
from operator import mod
from .base.abwrapper import ActionWrapper


class ClickGameWrapper(ActionWrapper):

    def __init__(self, x_gridsize=10, y_gridsize=10):
        self.x_gridsize = int(x_gridsize)
        self.y_gridsize = int(y_gridsize)

        self.named_actions = []

        for i in range (self.x_gridsize):
            self.named_actions.append("x"+str(i))

        for i in range (self.y_gridsize):
            self.named_actions.append("y"+str(i))

        self.multi_output_ranges = [0, self.x_gridsize, self.x_gridsize+self.y_gridsize]

        self.action_indices = [idx for idx in range(len(self.named_actions))]

    def is_action_done(self):
        return True

    def reset(self):
        pass

    def get_actions(self):
        return self.action_indices

    def get_excluded_actions(self, obs):
        return []

    def get_action(self, action_idx, obs):
        x, y = action_idx

        adjusted_x = x - self.multi_output_ranges[0]
        adjusted_y = y - self.multi_output_ranges[1]

        #return sc2.no_op()
        return [adjusted_x, adjusted_y]

class ClickGameBDQWrapper(ActionWrapper):

    def __init__(self, x_gridsize=10, y_gridsize=10):
        self.x_gridsize = int(x_gridsize)
        self.y_gridsize = int(y_gridsize)

        self.named_actions = []

        for i in range (self.x_gridsize):
            self.named_actions.append("x"+str(i))

        for i in range (self.y_gridsize):
            self.named_actions.append("y"+str(i))

        self.multi_output_ranges = [0, self.x_gridsize, self.x_gridsize+self.y_gridsize]

        self.action_indices = [idx for idx in range(len(self.named_actions))]

    def is_action_done(self):
        return True

    def reset(self):
        pass

    def get_actions(self):
        return self.action_indices

    def get_excluded_actions(self, obs):
        return []

    def get_action(self, action_idx, obs):
        x, y = action_idx
        return [x, y]

class ClickGameDDQNWrapper(ActionWrapper):

    def __init__(self, size_x=10, size_y=10):
        self.size_x = size_x
        self.size_y = size_y
        self.size = size_x * size_y
        self.named_actions = []

        for i in range (self.size):
            self.named_actions.append(str(i))

        self.action_indices = [idx for idx in range(len(self.named_actions))]

    def is_action_done(self):
        return True

    def reset(self):
        pass

    def get_actions(self):
        return self.action_indices

    def get_excluded_actions(self, obs):
        return []

    def get_action(self, action_idx, obs):
        x = action_idx % self.size_x
        y = floor(action_idx/self.size_x)
        return [x, y]

class ClickGameDeconvWrapper(ActionWrapper):

    def __init__(self, size_x=10, size_y=10):
        self.size_x = size_x
        self.size_y = size_y
        self.size = size_x * size_y
        self.named_actions = []

        for i in range (self.size):
            self.named_actions.append(str(i))

        self.action_indices = [idx for idx in range(len(self.named_actions))]

    def is_action_done(self):
        return True

    def reset(self):
        pass

    def get_actions(self):
        return self.action_indices

    def get_action_shape(self):
        return (self.size_x, self.size_y)

    def get_excluded_actions(self, obs):
        return []

    def get_action(self, action_idx, obs):
        x, y = action_idx
        return [x, y]