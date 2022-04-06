from .collectables import CollectablesGeneralizedRewardBuilder


class FindAndDefeatGeneralizedRewardBuilder(CollectablesGeneralizedRewardBuilder):

    def get_drts_reward(self, obs):
        enemy = 1
        archer = 7

        current = self.get_drts_number_of_specific_units(obs, enemy, archer)
        prev = self.get_drts_number_of_specific_units(self.previous_state, enemy, archer)
        # print(">>>>>> CURR NUMBER OF ENEMY ARCHERS: {}".format(current))

        rwdA = (current - prev) * 1000

        player = 0
        archer = 7

        current = self.get_drts_number_of_specific_units(obs, player, archer)
        prev = self.get_drts_number_of_specific_units(self.previous_state, player, archer)
        # print(">>>>>> CURR NUMBER OF PLAYER ARCHERS: {}".format(current))

        rwdB = (current - prev) * 1000

        # print(">>>>>> REWARD WAS: {}".format(rwdB - rwdA))

        return rwdB - rwdA

    def get_sc2_reward(self, obs):
        current = self.get_sc2_number_of_zerglings(obs)
        prev = self.get_sc2_number_of_zerglings(self.previous_state)

        rwdA = (current - prev) * 1000

        current = self.get_sc2_number_of_marines(obs)
        prev = self.get_sc2_number_of_marines(self.previous_state)

        rwdB = (current - prev) * 1000

        return rwdB - rwdA
