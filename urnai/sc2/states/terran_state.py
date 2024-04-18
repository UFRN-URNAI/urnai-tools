import math

import numpy as np
from pysc2.env import sc2_env
from pysc2.lib import features, units

from urnai.sc2.states.states_utils import get_my_units_amount
from urnai.states.state_base import StateBase


class TerranState(StateBase):

    def __init__(self, grid_size=4):
        self.grid_size = grid_size
        # Size of the state returned
        # 22: number of data added to the state (number of minerals, army_count, etc)
        # 2 * 4 ** 2: 2 grids of size 4 x 4, representing enemy and player units
        self.dimension = int(22 + 2 * (self.grid_size ** 2))
        self.player_race = sc2_env.Race.terran
        self.reset()

    def update(self, obs):
        new_state = [
            # Adds general information from the player.
            obs.player.minerals / 6000,
            obs.player.vespene / 6000,
            obs.player.food_cap / 200,
            obs.player.food_used / 200,
            obs.player.food_army / 200,
            obs.player.food_workers / 200,
            (obs.player.food_cap - obs.player.food_used) / 200,
            obs.player.army_count / 200,
            obs.player.idle_worker_count / 200,
            # Adds information related to player's Terran units/buildings.
            get_my_units_amount(obs, units.Terran.CommandCenter) +
            get_my_units_amount(obs, units.Terran.OrbitalCommand) +
            get_my_units_amount(obs, units.Terran.PlanetaryFortress) / 2,
            get_my_units_amount(obs, units.Terran.SupplyDepot) / 18,
            get_my_units_amount(obs, units.Terran.Refinery) / 4,
            get_my_units_amount(obs, units.Terran.EngineeringBay),
            get_my_units_amount(obs, units.Terran.Armory),
            get_my_units_amount(obs, units.Terran.MissileTurret) / 4,
            get_my_units_amount(obs, units.Terran.SensorTower)/1,
            get_my_units_amount(obs, units.Terran.Bunker)/4,
            get_my_units_amount(obs, units.Terran.FusionCore),
            get_my_units_amount(obs, units.Terran.GhostAcademy),
            get_my_units_amount(obs, units.Terran.Barracks) / 3,
            get_my_units_amount(obs, units.Terran.Factory) / 2,
            get_my_units_amount(obs, units.Terran.Starport) / 2
        ]

        new_state = append_player_and_enemy_grids(obs, new_state, self.grid_size)

        final_state = np.expand_dims(new_state, axis=0)

        self.state = final_state
        return final_state

    @property
    def state(self):
        return self.state

    @property
    def dimension(self):
        return self.dimension

    def reset(self):
        self.state = None

def append_player_and_enemy_grids(obs, new_state, grid_size):
    """ Instead of making a vector for all coordnates on the map, we'll
    discretize our enemy space and use a grid to store enemy positions
    by marking a square as 1 if there's any enemy on it."""
    enemy_grid = np.zeros((grid_size, grid_size))
    player_grid = np.zeros((grid_size, grid_size))

    enemy_units = [unit for unit in obs.raw_units if
                    unit.alliance == features.PlayerRelative.ENEMY]
    player_units = [unit for unit in obs.raw_units if
                    unit.alliance == features.PlayerRelative.SELF]

    for enemy_index in range(0, len(enemy_units)):
        #TODO: Fix the "64" number, probably should be the size of the map
        y = int(math.ceil((enemy_units[enemy_index].x + 1) / (64 / grid_size)))
        x = int(math.ceil((enemy_units[enemy_index].y + 1) / (64 / grid_size)))
        enemy_grid[x - 1][y - 1] += 1

    for player_index in range(0, len(player_units)):
        #TODO: Fix the "64" number, probably should be the size of the map
        y = int(math.ceil((player_units[player_index].x + 1) / (64 / grid_size)))
        x = int(math.ceil((player_units[player_index].y + 1) / (64 / grid_size)))
        player_grid[x - 1][y - 1] += 1

    # Normalizing the values to always be between 0 and 1
    # (since the max amount of units in SC2 is 200)
    enemy_grid = enemy_grid / 200
    player_grid = player_grid / 200

    new_state.extend(enemy_grid.flatten())
    new_state.extend(player_grid.flatten())

    return new_state
