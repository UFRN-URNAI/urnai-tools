import numpy as np
from pysc2.env import sc2_env
from pysc2.lib import units

from urnai.sc2.states.state_constants import (
    MAX_MINERALS,
    MAX_UNITS,
    MAX_VESPENE,
)
from urnai.sc2.states.states_utils import (
    append_player_and_enemy_grids,
    get_my_raw_units_amount,
)
from urnai.states.state_base import StateBase


class ProtossState(StateBase):

    def __init__(
        self,
        grid_size: int = 4,
        use_raw_units: bool = True,
        raw_resolution: int = 64,
    ):
        self.grid_size = grid_size
        self.player_race = sc2_env.Race.protoss
        self.use_raw_units = use_raw_units
        self.raw_resolution = raw_resolution
        self.reset()

    def update(self, obs):
        new_state = [
            # Adds general information from the player.
            obs.player.minerals / MAX_MINERALS,
            obs.player.vespene / MAX_VESPENE,
            obs.player.food_cap / MAX_UNITS,
            obs.player.food_used / MAX_UNITS,
            obs.player.food_army / MAX_UNITS,
            obs.player.food_workers / MAX_UNITS,
            (obs.player.food_cap - obs.player.food_used) / MAX_UNITS,
            obs.player.army_count / MAX_UNITS,
            obs.player.idle_worker_count / MAX_UNITS,
        ]

        if self.use_raw_units:
            new_state.extend(
                [
                    # Adds information related to player's Protoss units/buildings.
                    get_my_raw_units_amount(obs, units.Protoss.Nexus),
                    get_my_raw_units_amount(obs, units.Protoss.Pylon),
                    get_my_raw_units_amount(obs, units.Protoss.Assimilator),
                    get_my_raw_units_amount(obs, units.Protoss.Forge),
                    get_my_raw_units_amount(obs, units.Protoss.Gateway),
                    get_my_raw_units_amount(obs, units.Protoss.CyberneticsCore),
                    get_my_raw_units_amount(obs, units.Protoss.PhotonCannon),
                    get_my_raw_units_amount(obs, units.Protoss.RoboticsFacility),
                    get_my_raw_units_amount(obs, units.Protoss.Stargate),
                    get_my_raw_units_amount(obs, units.Protoss.TwilightCouncil),
                    get_my_raw_units_amount(obs, units.Protoss.RoboticsBay),
                    get_my_raw_units_amount(obs, units.Protoss.TemplarArchive),
                    get_my_raw_units_amount(obs, units.Protoss.DarkShrine),
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
