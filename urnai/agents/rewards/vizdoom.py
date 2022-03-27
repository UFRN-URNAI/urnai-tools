from .abreward import RewardBuilder


class VizDoomDefaultReward(RewardBuilder):

    def get_reward(self, obs, reward, done):
        return reward


class VizDoomHealthGatheringReward(RewardBuilder):

    KILLCOUNT = 0
    ITEMCOUNT = 1
    SECRETCOUNT = 2
    DEATHCOUNT = 3
    HITCOUNT = 4
    HITS_TAKEN = 5
    DAMAGECOUNT = 6
    DAMAGE_TAKEN = 7
    HEALTH = 8
    ARMOR = 9
    DEAD = 10
    ATTACK_READY = 11
    SELECTED_WEAPON = 12
    SELECTED_WEAPON_AMMO = 13
    POSITION_X = 14
    POSITION_Y = 15
    POSITION_Z = 16
    GENERAL_REWARD = 17

    METHOD_CUMULATIVE = 'cumulative'
    METHOD_DIFFERENCE = 'difference'
    METHOD_POSITIVE_ONLY = 'positive_only'
    METHOD_POSITIVE_ONLY_MINUS_ONE = 'positive_only_minus_one'
    METHOD_POSITIVE_ONLY_WEIGHTENED = 'positive_only_weightened'
    METHOD_POSITIVE_ONLY_WEIGHTENED_MINUS_ONE = 'positive_only_weightened_minus_one'

    def __init__(self, method):
        self.method = method
        self.prev_health = 0

    def get_reward(self, obs, reward, done):
        r = 0

        if self.method == VizDoomHealthGatheringReward.METHOD_CUMULATIVE:
            r += obs.game_variables[VizDoomHealthGatheringReward.HEALTH]
        elif self.method == VizDoomHealthGatheringReward.METHOD_DIFFERENCE:
            r += obs.game_variables[VizDoomHealthGatheringReward.HEALTH] - self.prev_health
            self.prev_health = obs.game_variables[VizDoomHealthGatheringReward.HEALTH]
        elif self.method == VizDoomHealthGatheringReward.METHOD_POSITIVE_ONLY:
            r += obs.game_variables[VizDoomHealthGatheringReward.HEALTH] - self.prev_health
            self.prev_health = obs.game_variables[VizDoomHealthGatheringReward.HEALTH]
            if r < 0:
                r = 0
        elif self.method == VizDoomHealthGatheringReward.METHOD_POSITIVE_ONLY_MINUS_ONE:
            r += obs.game_variables[VizDoomHealthGatheringReward.HEALTH] - self.prev_health
            self.prev_health = obs.game_variables[VizDoomHealthGatheringReward.HEALTH]
            if r < 0:
                r = -1
        elif self.method == VizDoomHealthGatheringReward.METHOD_POSITIVE_ONLY_WEIGHTENED:
            r += obs.game_variables[VizDoomHealthGatheringReward.HEALTH] - self.prev_health
            self.prev_health = obs.game_variables[VizDoomHealthGatheringReward.HEALTH] * 1000
            if r < 0:
                r = 0
        elif self.method == VizDoomHealthGatheringReward.METHOD_POSITIVE_ONLY_WEIGHTENED_MINUS_ONE:
            r += obs.game_variables[VizDoomHealthGatheringReward.HEALTH] - self.prev_health
            self.prev_health = obs.game_variables[VizDoomHealthGatheringReward.HEALTH] * 1000
            if r < 0:
                r = -1

        return r


class VizDoom2CustomReward(RewardBuilder):
    KILLCOUNT = 0
    ITEMCOUNT = 1
    SECRETCOUNT = 2
    DEATHCOUNT = 3
    HITCOUNT = 4
    HITS_TAKEN = 5
    DAMAGECOUNT = 6
    DAMAGE_TAKEN = 7
    HEALTH = 8
    ARMOR = 9
    DEAD = 10
    ATTACK_READY = 11
    SELECTED_WEAPON = 12
    SELECTED_WEAPON_AMMO = 13
    POSITION_X = 14
    POSITION_Y = 15
    POSITION_Z = 16
    GENERAL_REWARD = 17

    def get_reward(self, obs, reward, done):
        r = 0

        r += 10 * obs.game_variables[VizDoom2CustomReward.HITCOUNT]
        r += 50 * obs.game_variables[VizDoom2CustomReward.KILLCOUNT]
        r += -10 * obs.game_variables[VizDoom2CustomReward.DAMAGE_TAKEN]
        r += -100 * obs.game_variables[VizDoom2CustomReward.DEAD]
        r += 10 * obs.game_variables[VizDoom2CustomReward.SELECTED_WEAPON_AMMO]

        return r
