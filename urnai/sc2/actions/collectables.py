from statistics import mean

from pysc2.env import sc2_env
from pysc2.lib import actions

from urnai.actions.action_space_base import ActionSpaceBase
from urnai.sc2.actions import sc2_actions_aux as scaux
from urnai.sc2.actions.sc2_action import SC2Action


class CollectablesActionSpace(ActionSpaceBase):

    def __init__(self):
        self.noaction = [actions.RAW_FUNCTIONS.no_op()]
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
        self.named_actions = None

    def is_action_done(self):
        # return len(self.pending_actions) == 0
        return True

    def reset(self):
        self.move_number = 0
        self.pending_actions = []

    def get_actions(self):
        return self.action_indices

    def get_excluded_actions(self, obs):
        return []

    def get_action(self, action_idx, obs):
        action = None
        if len(self.pending_actions) == 0:
            action = [actions.RAW_FUNCTIONS.no_op()]
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
            # if action_idx was None, this means that the actionwrapper
            # was not resetted properly, so I will reset it here
            # this is not the best way to fix this
            # but until we cannot find why the agent is
            # not resetting the action wrapper properly
            # i'm gonna leave this here
            self.reset()

    def move_left(self, obs):
        army = scaux.select_army(obs, sc2_env.Race.terran)
        xs = [unit.x for unit in army]
        ys = [unit.y for unit in army]

        new_army_x = int(mean(xs)) - self.hor_threshold
        new_army_y = int(mean(ys))

        for unit in army:
            self.pending_actions.append(
                SC2Action.run(actions.RAW_FUNCTIONS.Move_pt,
                              'now', unit.tag, [new_army_x, new_army_y]))

    def move_right(self, obs):
        army = scaux.select_army(obs, sc2_env.Race.terran)
        xs = [unit.x for unit in army]
        ys = [unit.y for unit in army]

        new_army_x = int(mean(xs)) + self.hor_threshold
        new_army_y = int(mean(ys))

        for unit in army:
            self.pending_actions.append(
                SC2Action.run(actions.RAW_FUNCTIONS.Move_pt, 
                              'now', unit.tag, [new_army_x, new_army_y]))

    def move_down(self, obs):
        army = scaux.select_army(obs, sc2_env.Race.terran)
        xs = [unit.x for unit in army]
        ys = [unit.y for unit in army]

        new_army_x = int(mean(xs))
        new_army_y = int(mean(ys)) + self.ver_threshold

        for unit in army:
            self.pending_actions.append(
                SC2Action.run(actions.RAW_FUNCTIONS.Move_pt,
                              'now', unit.tag, [new_army_x, new_army_y]))

    def move_up(self, obs):
        army = scaux.select_army(obs, sc2_env.Race.terran)
        xs = [unit.x for unit in army]
        ys = [unit.y for unit in army]

        new_army_x = int(mean(xs))
        new_army_y = int(mean(ys)) - self.ver_threshold

        for unit in army:
            self.pending_actions.append(
                SC2Action.run(actions.RAW_FUNCTIONS.Move_pt,
                              'now', unit.tag, [new_army_x, new_army_y]))

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
