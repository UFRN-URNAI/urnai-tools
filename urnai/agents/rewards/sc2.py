"""
This file is a repository with reward classes for all StarCraft 2
games/minigames we've solved.
"""
from email.policy import default
import math
import numpy as np
from cmath import sqrt
from pysc2.lib import units
from urnai.agents.actions.sc2 import unit_exists
from urnai.agents.rewards.abreward import RewardBuilder


class SparseReward(RewardBuilder):
    def get_reward(self, obs, reward, done):
        """Always returns 0, unless the game has ended."""
        if not done:
            return 0
        return reward


class GeneralReward(RewardBuilder):
    """
    This reward function gives a reward value every step. The reward is a constant -1 value
    as long as the agent doesn't do anything. If the agent does something "positive" such as
    creating more army, workers, structures or killing units, the reward value receives a positive
    value proportional to the score change of those categories. Every step the current reward is
    then reset to -1.
    """

    def __init__(self):
        self.reward = 0

        self.last_own_worker_count = 0
        self.last_own_army_count = 0
        self.last_structures_score = 0
        self.last_killed_units_score = 0
        self.last_killed_structures_score = 0

    # When the episode is over, the values we use to compute our reward should be reset.
    def reset(self):
        self.last_own_worker_count = 0
        self.last_own_army_count = 0
        self.last_structures_score = 0
        self.last_killed_units_score = 0
        self.last_killed_structures_score = 0

    def get_reward(self, obs, reward, done):
        currentscore = -1
        currentscore += (obs.player.food_army - self.last_own_army_count) * 50
        currentscore += (obs.player.food_workers - self.last_own_worker_count) * 25
        currentscore += obs.score_cumulative.total_value_structures - self.last_structures_score
        currentscore += (obs.score_cumulative.killed_value_units - self.last_killed_units_score)
        currentscore += (obs.score_cumulative.killed_value_structures -
                         self.last_killed_structures_score) * 2

        self.last_own_army_count = obs.player.food_army
        self.last_own_worker_count = obs.player.food_workers
        self.last_killed_units_score = obs.score_cumulative.killed_value_units
        self.last_killed_structures_score = obs.score_cumulative.killed_value_structures
        self.last_structures_score = obs.score_cumulative.total_value_structures

        self.reward = currentscore

        if done:
            self.reset()

        return self.reward


class KilledUnitsReward(RewardBuilder):
    """
    A simple reward function in which the reward value is the difference in score points from
    killing enemy units or structures at the current time step compared the previous time step.
    """

    def __init__(self):

        # Properties keep track of the change of values used in our reward system
        self._previous_killed_unit_score = 0
        self._previous_killed_building_score = 0

    # When the episode is over, the values we use to compute our reward should be reset.
    def reset(self):
        self._previous_killed_unit_score = 0
        self._previous_killed_building_score = 0

    def get_reward(self, obs, reward, done):

        new_reward = 0

        new_reward += (obs.score_cumulative.killed_value_units - self._previous_killed_unit_score)
        new_reward += (
                obs.score_cumulative.killed_value_structures - self._previous_killed_building_score)

        self._previous_killed_unit_score = obs.score_cumulative.killed_value_units
        self._previous_killed_building_score = obs.score_cumulative.killed_value_structures

        if done:
            self.reset()

        if reward == 1:
            new_reward = 5000

        return new_reward


class KilledUnitsRewardBoosted(RewardBuilder):
    """
    A reward class similar to KilledUnitsReward but with added reward bonuses for constructing
    certain structures
    and training certain units. For example, constructing a Barrack, Factory or Starport is
    worth 300 reward points.
    """

    def __init__(self):

        self.construction_reward = 1000
        self.big_unit_reward = 300
        self.small_unit_reward = 200

        # Properties keep track of the change of values used in our reward system
        self._previous_killed_unit_score = 0
        self._previous_killed_building_score = 0

        self._has_barracks = False
        self._has_factory = False
        self._has_starport = False

        self._trained_tank = False
        self._trained_medivac = False
        self._trained_hellion = False

    # When the episode is over, the values we use to compute our reward should be reset.
    def reset(self):
        self._previous_killed_unit_score = 0
        self._previous_killed_building_score = 0

        self._has_barracks = False
        self._has_factory = False
        self._has_starport = False

        self._trained_tank = False
        self._trained_medivac = False
        self._trained_hellion = False

    def get_reward(self, obs, reward, done):

        new_reward = 0

        if unit_exists(obs, units.Terran.Barracks) and not self._has_barracks:
            new_reward += self.construction_reward
            self._has_barracks = True

        if unit_exists(obs, units.Terran.Factory) and not self._has_factory:
            new_reward += self.construction_reward
            self._has_factory = True

        if unit_exists(obs, units.Terran.Starport) and not self._has_starport:
            new_reward += self.construction_reward
            self._has_starport = True

        if unit_exists(obs, units.Terran.SiegeTank) and not self._trained_tank:
            new_reward += self.big_unit_reward
            self._trained_tank = True

        if unit_exists(obs, units.Terran.Medivac) and not self._trained_medivac:
            new_reward += self.big_unit_reward
            self._trained_medivac = True

        if unit_exists(obs, units.Terran.Hellion) and not self._trained_hellion:
            new_reward += self.small_unit_reward
            self._trained_hellion = True

        new_reward += (obs.score_cumulative.killed_value_units - self._previous_killed_unit_score)
        new_reward += (
                obs.score_cumulative.killed_value_structures - self._previous_killed_building_score)

        self._previous_killed_unit_score = obs.score_cumulative.killed_value_units
        self._previous_killed_building_score = obs.score_cumulative.killed_value_structures

        if done:
            self.reset()

        if reward == 1:
            new_reward = 5000

        return new_reward


class MoveToBeaconProximity(RewardBuilder):
    def __init__(self, default=False, boost=0.01):
        self.previous_distance = 100
        self.default = default
        self.boost = boost

    def get_reward(self, obs, reward, done):
        if not self.default:
            marine_and_beacon = [unit for unit in obs.raw_units if unit.unit_type == units.Terran.Marine or unit.unit_type == 317]
            x1 = marine_and_beacon[0].x
            y1 = marine_and_beacon[0].y
            x2 = marine_and_beacon[1].x
            y2 = marine_and_beacon[1].y

            curr_distance = math.hypot(x2-x1, y2-y1)

            if curr_distance < self.previous_distance:
                reward += self.boost

            self.previous_distance = curr_distance

        return reward

class MoveToBeaconDirection(RewardBuilder):
    def __init__(self, boost=0.01, add_distance = False):
        self.previous_distance = 100
        self.previous_x = 0
        self.previous_y = 0
        self.boost = boost
        self.add_distance = add_distance

    def get_reward(self, obs, reward, done):
        marine = [unit for unit in obs.raw_units if unit.unit_type == units.Terran.Marine][0]
        beacon = [unit for unit in obs.raw_units if unit.unit_type == 317][0]
        
        # Defining points for angle calculation, p1 = previous marine pos, p2 = current marine pos, q = beacon pos
        p1 = np.array([self.previous_x, self.previous_y])
        p2 = np.array([marine.x, marine.y])
        q  = np.array([beacon.x, beacon.y])

        # Calculating the angle from the marine direction vector and the beacon position using the vector dot product definition
        v1 = p2-p1
        v2 = q-p1
        angle = np.lib.scimath.arccos( np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)) )

        # angle goes from 0 (marine moving in the direction of the beacon)
        # to pi (marine moving in the exact oposite direction of the beacon)
        # we want to map that so when the marine is moving towards the beacon we get 1, and when its moving opposite we get -1
        reward_angle = (angle - (math.pi/2)) * -1 / (math.pi/2)

        # If the marine pos doesn't change our calculations break and we get angle = NaN, so this solves that
        if math.isnan(reward_angle):
            reward_angle = 0

        # Multiplying our angled_reward (goes from -1 to 1) by our boost factor (0.01 by default)
        reward += reward_angle*self.boost

        if self.add_distance:
            curr_distance = math.hypot(beacon.x-marine.x, beacon.y-marine.y)
            # Adding half of our boost factor to the reward if we're closing the distance from the marine to the beacon
            if curr_distance < self.previous_distance:
                    reward += self.boost/2

            self.previous_distance = curr_distance

        self.previous_x = marine.x
        self.previous_y = marine.y

        return reward

"""
Ideas for new reward builders or improvements for current ones:

    - Use utils.sc2_utils.get_fog_of_war_percentage() as a part of a reward value.
    Ex: if the agent explored 25% of the map give some reward, if explored 50% give another
    reward, etc.

    - Try out GeneralReward again but without the constant -1, just either 0 or positive reward,
    and try and balance different scores more (for instance, food_army score is multiplied by 50
    and that may not be optimal)

    - Create a reward for researching technology, or for creating techlabs in factory, starport etc
    so we can encourage the agent to train more complex units and have better army composition
"""
