import numpy as np
from pysc2.env import sc2_env
from pysc2.lib import units

from urnai.constants import SC2Constants
from urnai.sc2.states.utils import (
    append_player_and_enemy_grids,
    create_raw_units_amount_dict,
)
from urnai.states.state_base import StateBase


class ZergState(StateBase):

    def __init__(
        self,
        grid_size: int = 4,
        use_raw_units: bool = True,
        raw_resolution: int = 64,
    ):
        self.grid_size = grid_size
        self.player_race = sc2_env.Race.zerg
        self.use_raw_units = use_raw_units
        self.raw_resolution = raw_resolution
        self.reset()

    def update(self, obs):
        new_state = [
            # Adds general information from the player.
            obs.player.minerals / SC2Constants.MAX_MINERALS,
            obs.player.vespene / SC2Constants.MAX_VESPENE,
            obs.player.food_cap / SC2Constants.MAX_UNITS,
            obs.player.food_used / SC2Constants.MAX_UNITS,
            obs.player.food_army / SC2Constants.MAX_UNITS,
            obs.player.food_workers / SC2Constants.MAX_UNITS,
            (obs.player.food_cap - obs.player.food_used) / SC2Constants.MAX_UNITS,
            obs.player.army_count / SC2Constants.MAX_UNITS,
            obs.player.idle_worker_count / SC2Constants.MAX_UNITS,
        ]

        if self.use_raw_units:
            raw_units_amount_dict = create_raw_units_amount_dict(
                obs, sc2_env.features.PlayerRelative.SELF)
            new_state.extend(
                [
                    # Adds information related to player's Zerg units/buildings.
                    raw_units_amount_dict[units.Zerg.BanelingNest],
                    raw_units_amount_dict[units.Zerg.EvolutionChamber],
                    raw_units_amount_dict[units.Zerg.Extractor],
                    raw_units_amount_dict[units.Zerg.Hatchery],
                    raw_units_amount_dict[units.Zerg.HydraliskDen],
                    raw_units_amount_dict[units.Zerg.InfestationPit],
                    raw_units_amount_dict[units.Zerg.LurkerDen],
                    raw_units_amount_dict[units.Zerg.NydusNetwork],
                    raw_units_amount_dict[units.Zerg.RoachWarren],
                    raw_units_amount_dict[units.Zerg.SpawningPool],
                    raw_units_amount_dict[units.Zerg.SpineCrawler],
                    raw_units_amount_dict[units.Zerg.Spire],
                    raw_units_amount_dict[units.Zerg.SporeCrawler],
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
