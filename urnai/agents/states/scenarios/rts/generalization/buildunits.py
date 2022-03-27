from pysc2.lib import units as sc2units
import urnai.agents.actions.sc2 as sc2aux
from urnai.utils.constants import RTSGeneralization

from .defeatenemies import DefeatEnemiesGeneralizedStatebuilder


class BuildUnitsGeneralizedStatebuilder(DefeatEnemiesGeneralizedStatebuilder):

    def __init__(self, method=RTSGeneralization.STATE_MAP):
        super().__init__(method=method)
        self.non_spatial_maximums = [
            RTSGeneralization.STATE_MAXIMUM_GOLD_OR_MINERALS,
            RTSGeneralization.MAXIMUM_NUMBER_OF_FARM_OR_SUPPLY_DEPOT,
            RTSGeneralization.MAXIMUM_NUMBER_OF_BARRACKS,
            RTSGeneralization.MAXIMUM_NUMBER_OF_ARCHERS_MARINES,
        ]
        self.non_spatial_minimums = [0, 0, 0, 0]
        self.non_spatial_state = [0, 0, 0, 0]

    def build_non_spatial_sc2_state(self, obs):
        self.non_spatial_state[0] = obs.player.minerals
        self.non_spatial_state[1] = sc2aux.get_my_units_amount(obs, sc2units.Terran.SupplyDepot)
        self.non_spatial_state[2] = sc2aux.get_my_units_amount(obs, sc2units.Terran.Barracks)
        self.non_spatial_state[3] = sc2aux.get_my_units_amount(obs, sc2units.Terran.Marine)
        self.normalize_non_spatial_list()
        return self.non_spatial_state

    def build_non_spatial_drts_state(self, obs):
        player = 0
        farm = 6
        barracks = 4
        archer = 7

        self.non_spatial_state[0] = obs['players'][player].gold
        self.non_spatial_state[1] = len(self.get_drts_player_specific_type_units(obs, player, farm))
        self.non_spatial_state[2] = len(
            self.get_drts_player_specific_type_units(obs, player, barracks))
        self.non_spatial_state[3] = len(
            self.get_drts_player_specific_type_units(obs, player, archer))
        self.normalize_non_spatial_list()
        return self.non_spatial_state
