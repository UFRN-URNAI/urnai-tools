import numpy as np
from pysc2.env import sc2_env
from pysc2.lib import units

from urnai.sc2.states.starcraft2_state import StarCraft2State
from urnai.sc2.states.utils import create_raw_units_amount_dict


class ProtossState(StarCraft2State):

    def __init__(
        self,
        grid_size: int = 4,
        use_raw_units: bool = True,
        raw_resolution: int = 64,
    ):
        super().__init__(grid_size, use_raw_units, raw_resolution)
        self.player_race = sc2_env.Race.protoss

    def update(self, obs):
        state = super().update(obs)

        if self.use_raw_units:
            raw_units_amount_dict = create_raw_units_amount_dict(
                obs, sc2_env.features.PlayerRelative.SELF)
            units_amount_info = [
                raw_units_amount_dict[units.Protoss.Nexus],
                raw_units_amount_dict[units.Protoss.Pylon],
                raw_units_amount_dict[units.Protoss.Assimilator],
                raw_units_amount_dict[units.Protoss.Forge],
                raw_units_amount_dict[units.Protoss.Gateway],
                raw_units_amount_dict[units.Protoss.CyberneticsCore],
                raw_units_amount_dict[units.Protoss.PhotonCannon],
                raw_units_amount_dict[units.Protoss.RoboticsFacility],
                raw_units_amount_dict[units.Protoss.Stargate],
                raw_units_amount_dict[units.Protoss.TwilightCouncil],
                raw_units_amount_dict[units.Protoss.RoboticsBay],
                raw_units_amount_dict[units.Protoss.TemplarArchive],
                raw_units_amount_dict[units.Protoss.DarkShrine],
            ]
            state = np.squeeze(state)
            state = np.append(state, units_amount_info)
            self._dimension = len(state)
            state = np.expand_dims(state, axis=0)
            self._state = state

        return state