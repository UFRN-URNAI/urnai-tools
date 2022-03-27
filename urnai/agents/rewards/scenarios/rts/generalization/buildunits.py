from utils.constants import RTSGeneralization
from utils.reporter import Reporter as rp

from .defeatenemies import DefeatEnemiesGeneralizedRewardBuilder


class BuildUnitsGeneralizedRewardBuilder(DefeatEnemiesGeneralizedRewardBuilder):
    LAST_CHOSEN_ACTION = -1
    ACTION_DO_NOTHING = 7
    ACTION_BUILD_SUPPLY_DEPOT = 8
    ACTION_BUILD_BARRACK = 9
    ACTION_BUILD_MARINE = 10

    def get_drts_reward(self, obs):
        player = 0
        footman = 5
        farm = 6
        barracks = 4
        build_farm = RTSGeneralization.ACTION_DRTS_BUILD_FARM
        build_barrack = RTSGeneralization.ACTION_DRTS_BUILD_BARRACK
        build_footman = RTSGeneralization.ACTION_DRTS_BUILD_FOOTMAN
        # do_nothing = RTSGeneralization.ACTION_DRTS_DO_NOTHING

        current = self.get_drts_number_of_specific_units(obs, player, farm)
        prev = self.get_drts_number_of_specific_units(self.previous_state, player, farm)
        farm_amount_curr = (current - prev)

        current = self.get_drts_number_of_specific_units(obs, player, barracks)
        prev = self.get_drts_number_of_specific_units(self.previous_state, player, barracks)
        barracks_amount_curr = (current - prev)

        current = self.get_drts_number_of_specific_units(obs, player, footman)
        prev = self.get_drts_number_of_specific_units(self.previous_state, player, footman)
        footman_amount_curr = (current - prev)

        negative_rwd = 0
        chosen_action = BuildUnitsGeneralizedRewardBuilder.LAST_CHOSEN_ACTION
        if chosen_action > -1:
            farm_number = self.get_drts_number_of_specific_units(obs, player, farm)
            barracks_amount = self.get_drts_number_of_specific_units(obs, player, barracks)
            gold_amount = obs['players'][0].gold
            if chosen_action == build_farm:
                if farm_number > 7 or gold_amount < 500:
                    negative_rwd = -10
            elif chosen_action == build_barrack:
                if farm_number <= 0 or gold_amount < 700:
                    negative_rwd = -10
            elif chosen_action == build_footman:
                if barracks_amount <= 0 or gold_amount < 600:
                    negative_rwd = -10
            # elif chosen_action == do_nothing:
            #    negative_rwd = -1

        # rwd = negative_rwd + rwdB + rwdC
        rp.report(
            """
Calculated reward is: {},
composed of:
farm_amount: {},
barracks_amount: {},
footman_amount: {},
negative_rdw: {}
            """.format(
                negative_rwd + farm_amount_curr + barracks_amount_curr * 10
                + footman_amount_curr * 100,
                farm_amount_curr,
                barracks_amount_curr * 10,
                footman_amount_curr * 100,
                negative_rwd,
            ),
            verbosity_lvl=1,
        )
        if farm_amount_curr < 0 or barracks_amount_curr < 0 or footman_amount_curr < 0:
            return 0
        else:
            rwd = negative_rwd + farm_amount_curr + barracks_amount_curr * 10 \
                  + footman_amount_curr * 100
            return rwd

    def get_sc2_reward(self, obs):
        build_supply_depot = BuildUnitsGeneralizedRewardBuilder.ACTION_BUILD_SUPPLY_DEPOT
        build_barrack = BuildUnitsGeneralizedRewardBuilder.ACTION_BUILD_BARRACK
        build_marine = BuildUnitsGeneralizedRewardBuilder.ACTION_BUILD_MARINE
        # do_nothing = BuildUnitsGeneralizedRewardBuilder.ACTION_DO_NOTHING

        current = self.get_sc2_number_of_supply_depot(obs)
        prev = self.get_sc2_number_of_supply_depot(self.previous_state)
        supply_depot_amount_diff = (current - prev)

        current = self.get_sc2_number_of_barracks(obs)
        prev = self.get_sc2_number_of_barracks(self.previous_state)
        barracks_amount_diff = (current - prev)

        current = self.get_sc2_number_of_marines(obs)
        prev = self.get_sc2_number_of_marines(self.previous_state)
        marines_amount_diff = (current - prev)

        negative_rwd = 0
        chosen_action = BuildUnitsGeneralizedRewardBuilder.LAST_CHOSEN_ACTION
        if chosen_action > -1:
            supply_depot_amount = self.get_sc2_number_of_supply_depot(obs)
            barracks_amount = self.get_sc2_number_of_barracks(obs)
            minerals = obs.player.minerals
            if chosen_action == build_supply_depot:
                if supply_depot_amount > 7 or minerals < 100:
                    negative_rwd = -10
            elif chosen_action == build_barrack:
                if supply_depot_amount <= 0 or minerals < 150:
                    negative_rwd = -10
            elif chosen_action == build_marine:
                if barracks_amount <= 0 or minerals < 50:
                    negative_rwd = -10
            # elif chosen_action == do_nothing:
            #    negative_rwd = -1

        # rwd = negative_rwd + rwdB + rwdC
        rp.report(
            """
Calculated reward is: {},
composed of:
supply_depot_amount: {},
barracks_amount: {},
marines_amount: {},
negative_rdw: {}
            """.format(
                negative_rwd + supply_depot_amount_diff + barracks_amount_diff * 10
                + marines_amount_diff * 100,
                supply_depot_amount_diff,
                barracks_amount_diff * 10,
                marines_amount_diff * 100,
                negative_rwd,
            ),
            verbosity_lvl=1,
        )
        if supply_depot_amount_diff < 0 or barracks_amount_diff < 0 or marines_amount_diff < 0:
            return 0
        else:
            rwd = negative_rwd + supply_depot_amount_diff + barracks_amount_diff * 10 \
                  + marines_amount_diff * 100
            return rwd
        # if rwd > 0: return rwd
        # else: return 0
