import numpy as np
from pysc2.env import sc2_env
from pysc2.lib import units

from urnai.sc2.states.starcraft2_state import StarCraft2State
from urnai.sc2.states.utils import create_raw_units_amount_dict


class TerranState(StarCraft2State):

    def __init__(
        self,
        grid_size: int = 4,
        use_raw_units: bool = True,
        raw_resolution: int = 64,
    ):
        super().__init__(grid_size, use_raw_units, raw_resolution)
        self.player_race = sc2_env.Race.terran

    def update(self, obs):
        state = super().update(obs)

        if self.use_raw_units:
            raw_units_amount_dict = create_raw_units_amount_dict(
                obs, sc2_env.features.PlayerRelative.SELF)
            units_amount_info = [
                raw_units_amount_dict[units.Terran.CommandCenter]
                + raw_units_amount_dict[units.Terran.OrbitalCommand]
                + raw_units_amount_dict[units.Terran.PlanetaryFortress] / 2,
                raw_units_amount_dict[units.Terran.SupplyDepot] / 18,
                raw_units_amount_dict[units.Terran.Refinery] / 4,
                raw_units_amount_dict[units.Terran.EngineeringBay],
                raw_units_amount_dict[units.Terran.Armory],
                raw_units_amount_dict[units.Terran.MissileTurret] / 4,
                raw_units_amount_dict[units.Terran.SensorTower] / 1,
                raw_units_amount_dict[units.Terran.Bunker] / 4,
                raw_units_amount_dict[units.Terran.FusionCore],
                raw_units_amount_dict[units.Terran.GhostAcademy],
                raw_units_amount_dict[units.Terran.Barracks] / 3,
                raw_units_amount_dict[units.Terran.Factory] / 2,
                raw_units_amount_dict[units.Terran.Starport] / 2,
            ]
            state = np.squeeze(state)
            state = np.append(state, units_amount_info)
            self._dimension = len(state)
            state = np.expand_dims(state, axis=0)
            self._state = state

        return state