import math

import numpy as np
from pysc2.lib import features


def append_player_and_enemy_grids(obs, new_state, grid_size, raw_resolution):
    """ Instead of making a vector for all coordnates on the map, we'll
    discretize our enemy space and use a grid to store enemy positions
    by marking a square as 1 if there's any enemy on it."""
    enemy_grid = np.zeros((grid_size, grid_size))
    player_grid = np.zeros((grid_size, grid_size))

    enemy_units = [unit for unit in obs.raw_units if
                    unit.alliance == features.PlayerRelative.ENEMY]
    player_units = [unit for unit in obs.raw_units if
                    unit.alliance == features.PlayerRelative.SELF]
    raw_to_grid_ratio = raw_resolution / grid_size

    for enemy_index in range(0, len(enemy_units)):
        y = int(math.ceil((enemy_units[enemy_index].x + 1) / raw_to_grid_ratio))
        x = int(math.ceil((enemy_units[enemy_index].y + 1) / raw_to_grid_ratio))
        enemy_grid[x - 1][y - 1] += 1

    for player_index in range(0, len(player_units)):
        y = int(math.ceil((player_units[player_index].x + 1) / raw_to_grid_ratio))
        x = int(math.ceil((player_units[player_index].y + 1) / raw_to_grid_ratio))
        player_grid[x - 1][y - 1] += 1

    # Normalizing the values to always be between 0 and 1
    # (since the max amount of units in SC2 is 200)
    enemy_grid = enemy_grid / 200
    player_grid = player_grid / 200

    new_state.extend(enemy_grid.flatten())
    new_state.extend(player_grid.flatten())

    return new_state

def get_my_raw_units_amount(obs, unit_type):
    return len(get_raw_units_by_type(obs, unit_type, features.PlayerRelative.SELF))

def get_raw_units_by_type(obs, unit_type, alliance=features.PlayerRelative.SELF):
    return [unit for unit in obs.raw_units
            if unit.unit_type == unit_type
            and unit.alliance == alliance
            and unit.build_progress == 100]