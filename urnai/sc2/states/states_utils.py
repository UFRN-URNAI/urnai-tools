import math

import numpy as np
from pysc2.lib import features

from urnai.constants import SC2Constants


def append_player_and_enemy_grids(
        obs : list,
        new_state : list,
        grid_size : int,
        raw_resolution : int,
    ) -> list:
    new_state = append_grid(
        obs, new_state, grid_size, raw_resolution, features.PlayerRelative.ENEMY
    )
    new_state = append_grid(
        obs, new_state, grid_size, raw_resolution, features.PlayerRelative.SELF
    )
    return new_state

def append_grid(
        obs : list,
        new_state : list,
        grid_size : int, 
        raw_resolution : int,
        alliance : features.PlayerRelative,
    ) -> list:
    """ Instead of making a vector for all coordnates on the map, we'll
    discretize our unit space and use a grid to store unit positions
    by marking a square as 1 if there's any enemy/ally on it."""
    grid = np.zeros((grid_size, grid_size))

    units = [unit for unit in obs.raw_units if
             unit.alliance == alliance]
    raw_to_grid_ratio = raw_resolution / grid_size

    for index in range(0, len(units)):
        y = int(math.ceil((units[index].x + 1) / raw_to_grid_ratio))
        x = int(math.ceil((units[index].y + 1) / raw_to_grid_ratio))
        grid[x - 1][y - 1] += 1

    # Normalizing the values to always be between 0 and 1
    grid = grid / SC2Constants.MAX_UNITS

    new_state.extend(grid.flatten())

    return new_state

def get_my_raw_units_amount(obs : list, unit_type : int) -> int:
    return len(get_raw_units_by_type(obs, unit_type, features.PlayerRelative.SELF))

def get_raw_units_by_type(
        obs : list, 
        unit_type : int, 
        alliance : features.PlayerRelative = features.PlayerRelative.SELF,
    ) -> list:
    return [unit for unit in obs.raw_units
            if unit.unit_type == unit_type
            and unit.alliance == alliance
            and unit.build_progress == SC2Constants.BUILD_PROGRESS_COMPLETE]