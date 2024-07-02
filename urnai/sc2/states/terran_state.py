import numpy as np
from pysc2.env import sc2_env
from pysc2.lib import units

from urnai.sc2.states.states_utils import (
    append_player_and_enemy_grids,
    get_my_raw_units_amount,
)
from urnai.states.state_base import StateBase


class TerranState(StateBase):

    def __init__(
        self,
        grid_size: int = 4,
        use_raw_units: bool = True,
        raw_resolution: int = 64,
    ):
        self.grid_size = grid_size
        self.player_race = sc2_env.Race.terran
        self.use_raw_units = use_raw_units
        self.raw_resolution = raw_resolution
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
        ]

        if self.use_raw_units:
            new_state.extend(
                [
                    # Adds information related to player's Terran units/buildings.
                    get_my_raw_units_amount(obs, units.Terran.CommandCenter)
                    + get_my_raw_units_amount(obs, units.Terran.OrbitalCommand)
                    + get_my_raw_units_amount(obs, units.Terran.PlanetaryFortress) / 2,
                    get_my_raw_units_amount(obs, units.Terran.SupplyDepot) / 18,
                    get_my_raw_units_amount(obs, units.Terran.Refinery) / 4,
                    get_my_raw_units_amount(obs, units.Terran.EngineeringBay),
                    get_my_raw_units_amount(obs, units.Terran.Armory),
                    get_my_raw_units_amount(obs, units.Terran.MissileTurret) / 4,
                    get_my_raw_units_amount(obs, units.Terran.SensorTower) / 1,
                    get_my_raw_units_amount(obs, units.Terran.Bunker) / 4,
                    get_my_raw_units_amount(obs, units.Terran.FusionCore),
                    get_my_raw_units_amount(obs, units.Terran.GhostAcademy),
                    get_my_raw_units_amount(obs, units.Terran.Barracks) / 3,
                    get_my_raw_units_amount(obs, units.Terran.Factory) / 2,
                    get_my_raw_units_amount(obs, units.Terran.Starport) / 2,
                ]
            )
            new_state = append_player_and_enemy_grids(
                obs, new_state, self.grid_size, self.raw_resolution
            )

        self._dimension = len(new_state)
        final_state = np.expand_dims(new_state, axis=0)
        self._state = final_state
        return final_state

    @property
    def state(self):
        return self._state

    @property
    def dimension(self):
        return self._dimension

    def reset(self):
        self._state = None
        self._dimension = None
