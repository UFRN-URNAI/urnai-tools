import math

import numpy as np
from pysc2.env import sc2_env
from pysc2.lib import features, units

from urnai.states.state_base import StateBase


class TerranState(StateBase):
    
    def __init__(self, grid_size=4):

        self.grid_size = grid_size
        # Size of the state returned
        # 22: number of data added to the state (number of minerals, army_count, etc)
        # 2 * 4 ** 2: 2 grids of size 4 x 4, representing enemy and player units
        self._state_size = int(22 + 2 * (self.grid_size ** 2))
        self.player_race = sc2_env.Race.terran
        self.base_top_left = None
        self._state = None

    def update(self, obs):
        if obs.game_loop[0] < 80 and self.base_top_left is None:

            commandcenter = get_units_by_type(obs, units.Terran.CommandCenter)

            if len(commandcenter) > 0:
                townhall = commandcenter[0]
                self.player_race = sc2_env.Race.terran

            self.base_top_left = (townhall.x < 32)

        new_state = []
        # Adds general information from the player.
        new_state.append(obs.player.minerals / 6000)
        new_state.append(obs.player.vespene / 6000)
        new_state.append(obs.player.food_cap / 200)
        new_state.append(obs.player.food_used / 200)
        new_state.append(obs.player.food_army / 200)
        new_state.append(obs.player.food_workers / 200)
        new_state.append((obs.player.food_cap - obs.player.food_used) / 200)
        new_state.append(obs.player.army_count / 200)
        new_state.append(obs.player.idle_worker_count / 200)

        # Adds information related to player's Terran units/buildings.
        new_state.append(get_my_units_amount(obs, units.Terran.CommandCenter) +
                        get_my_units_amount(obs, units.Terran.OrbitalCommand) +
                        get_my_units_amount(obs, units.Terran.PlanetaryFortress) / 2)
        new_state.append(get_my_units_amount(obs, units.Terran.SupplyDepot) / 18)
        new_state.append(get_my_units_amount(obs, units.Terran.Refinery) / 4)
        new_state.append(get_my_units_amount(obs, units.Terran.EngineeringBay))
        new_state.append(get_my_units_amount(obs, units.Terran.Armory))
        new_state.append(get_my_units_amount(obs, units.Terran.MissileTurret) / 4)
        new_state.append(get_my_units_amount(obs, units.Terran.SensorTower)/1)
        new_state.append(get_my_units_amount(obs, units.Terran.Bunker)/4)
        new_state.append(get_my_units_amount(obs, units.Terran.FusionCore))
        new_state.append(get_my_units_amount(obs, units.Terran.GhostAcademy))
        new_state.append(get_my_units_amount(obs, units.Terran.Barracks) / 3)
        new_state.append(get_my_units_amount(obs, units.Terran.Factory) / 2)
        new_state.append(get_my_units_amount(obs, units.Terran.Starport) / 2)

        # Instead of making a vector for all coordnates on the map, we'll
        # discretize our enemy space
        # and use a 4x4 grid to store enemy positions by marking a square as 1 if
        # there's any enemy on it.

        enemy_grid = np.zeros((self.grid_size, self.grid_size))
        player_grid = np.zeros((self.grid_size, self.grid_size))

        enemy_units = [unit for unit in obs.raw_units if
                       unit.alliance == features.PlayerRelative.ENEMY]
        player_units = [unit for unit in obs.raw_units if
                        unit.alliance == features.PlayerRelative.SELF]

        for i in range(0, len(enemy_units)):
            y = int(math.ceil((enemy_units[i].x + 1) / 64 / self.grid_size))
            x = int(math.ceil((enemy_units[i].y + 1) / 64 / self.grid_size))
            enemy_grid[x - 1][y - 1] += 1

        for i in range(0, len(player_units)):
            y = int(math.ceil((player_units[i].x + 1) / (64 / self.grid_size)))
            x = int(math.ceil((player_units[i].y + 1) / (64 / self.grid_size)))
            player_grid[x - 1][y - 1] += 1

        if not self.base_top_left:
            enemy_grid = np.rot90(enemy_grid, 2)
            player_grid = np.rot90(player_grid, 2)

        # Normalizing the values to always be between 0 and 1
        # (since the max amount of units in SC2 is 200)
        enemy_grid = enemy_grid / 200
        player_grid = player_grid / 200

        new_state.extend(enemy_grid.flatten())
        new_state.extend(player_grid.flatten())
        final_state = np.expand_dims(new_state, axis=0)

        self._state = final_state
        return final_state

    @property
    def state(self):
        return self._state

    @property
    def dimension(self):
        return self._state_size

    def reset(self):
        self._state = None
        self.base_top_left = None

def get_my_units_amount(obs, unit_type):
    return len(get_units_by_type(obs, unit_type, features.PlayerRelative.SELF))

def get_units_by_type(obs, unit_type, alliance=features.PlayerRelative.SELF):
    return [unit for unit in obs.raw_units
            if unit.unit_type == unit_type
            and unit.alliance == alliance
            and unit.build_progress == 100]