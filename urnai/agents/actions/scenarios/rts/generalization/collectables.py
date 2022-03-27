from statistics import mean

from pysc2.env import sc2_env
from pysc2.lib import actions
from urnai.agents.actions import sc2 as scaux
from urnai.agents.actions.base.abwrapper import ActionWrapper


class CollectablesDeepRTSActionWrapper(ActionWrapper):
    def __init__(self):
        self.move_number = 0

        self.previousunit = 0
        self.nextunit = 1
        self.moveleft = 2
        self.moveright = 3
        self.moveup = 4
        self.movedown = 5
        self.moveupleft = 6
        self.moveupright = 7
        self.movedownleft = 8
        self.movedownright = 9
        self.attack = 10
        self.harvest = 11
        self.build0 = 12
        self.build1 = 13
        self.build2 = 14
        self.noaction = 15

        self.named_actions = ['move_left', 'move_right', 'move_up', 'move_down']
        self.excluded_actions = []

        self.final_actions = [self.moveleft, self.moveright, self.moveup, self.movedown]
        self.action_indices = range(len(self.final_actions))
        self.action_queue = []

    def get_player_units(self, player, obs):
        units = []
        for unit in obs['units']:
            if unit.get_player() == player:
                units.append(unit)

        return units

    def enqueue_action_for_player_units(self, obs, action):
        for i in range(len(self.get_player_units(obs['players'][0], obs))):
            self.action_queue.append(action)

    def get_action(self, action_idx, obs):
        action = None
        if len(self.action_queue) == 0:
            action = self.noaction
        else:
            action = self.action_queue.pop()
        self.solve_action(action_idx, obs)
        return action

    def solve_action(self, action_idx, obs):
        if action_idx is not None:
            if action_idx != self.noaction:
                i = action_idx
                if self.final_actions[i] == self.moveleft:
                    self.move_left(obs)
                elif self.final_actions[i] == self.moveright:
                    self.move_right(obs)
                elif self.final_actions[i] == self.moveup:
                    self.move_up(obs)
                elif self.final_actions[i] == self.movedown:
                    self.move_down(obs)
        else:
            # if action_idx was None, this means that the actionwrapper
            # was not resetted properly, so I will reset it here
            # this is not the best way to fix this
            # but until we cannot find why the agent is
            # not resetting the action wrapper properly
            # i'm gonna leave this here
            self.reset()

    def is_action_done(self):
        return True

    def reset(self):
        self.move_number = 0
        self.action_queue = []

    def get_actions(self):
        return self.action_indices

    def move_up(self, obs):
        self.enqueue_action_for_player_units(obs, self.moveup)

    def move_down(self, obs):
        self.enqueue_action_for_player_units(obs, self.movedown)

    def move_left(self, obs):
        self.enqueue_action_for_player_units(obs, self.moveleft)

    def move_right(self, obs):
        self.enqueue_action_for_player_units(obs, self.moveright)

    def get_excluded_actions(self, obs):
        return self.excluded_actions

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


class CollectablesStarcraftIIActionWrapper(ActionWrapper):

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
                actions.RAW_FUNCTIONS.Move_pt('now', unit.tag, [new_army_x, new_army_y]))

    def move_right(self, obs):
        army = scaux.select_army(obs, sc2_env.Race.terran)
        xs = [unit.x for unit in army]
        ys = [unit.y for unit in army]

        new_army_x = int(mean(xs)) + self.hor_threshold
        new_army_y = int(mean(ys))

        for unit in army:
            self.pending_actions.append(
                actions.RAW_FUNCTIONS.Move_pt('now', unit.tag, [new_army_x, new_army_y]))

    def move_down(self, obs):
        army = scaux.select_army(obs, sc2_env.Race.terran)
        xs = [unit.x for unit in army]
        ys = [unit.y for unit in army]

        new_army_x = int(mean(xs))
        new_army_y = int(mean(ys)) + self.ver_threshold

        for unit in army:
            self.pending_actions.append(
                actions.RAW_FUNCTIONS.Move_pt('now', unit.tag, [new_army_x, new_army_y]))

    def move_up(self, obs):
        army = scaux.select_army(obs, sc2_env.Race.terran)
        xs = [unit.x for unit in army]
        ys = [unit.y for unit in army]

        new_army_x = int(mean(xs))
        new_army_y = int(mean(ys)) - self.ver_threshold

        for unit in army:
            self.pending_actions.append(
                actions.RAW_FUNCTIONS.Move_pt('now', unit.tag, [new_army_x, new_army_y]))

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
