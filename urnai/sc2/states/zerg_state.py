import numpy as np
from pysc2.env import sc2_env
from pysc2.lib import units

from urnai.sc2.states.starcraft2_state import StarCraft2State
from urnai.sc2.states.utils import create_raw_units_amount_dict


class ZergState(StarCraft2State):

    def __init__(
        self,
        grid_size: int = 4,
        use_raw_units: bool = True,
        raw_resolution: int = 64,
    ):
        super().__init__(grid_size, use_raw_units, raw_resolution)
        self.player_race = sc2_env.Race.zerg

    def update(self, obs):
        state = super().update(obs)

        if self.use_raw_units:
            raw_units_amount_dict = create_raw_units_amount_dict(
                obs, sc2_env.features.PlayerRelative.SELF)
            units_amount_info = [
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
            state = np.squeeze(state)
            state = np.append(state, units_amount_info)
            self._dimension = len(state)
            state = np.expand_dims(state, axis=0)
            self._state = state

        return state