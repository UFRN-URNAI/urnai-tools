import numpy as np
import pysc2
from pysc2.env import sc2_env

sc2_races = {
    'terran': sc2_env.Race.terran,
    'protoss': sc2_env.Race.protoss,
    'zerg': sc2_env.Race.zerg,
    'random': sc2_env.Race.random}
sc2_difficulties = {
    'very_easy': sc2_env.Difficulty.very_easy,
    'easy': sc2_env.Difficulty.easy,
    'medium': sc2_env.Difficulty.medium,
    'medium_hard': sc2_env.Difficulty.medium_hard,
    'hard': sc2_env.Difficulty.hard,
    'harder': sc2_env.Difficulty.harder,
    'very_hard': sc2_env.Difficulty.very_hard,
    'cheat_vision': sc2_env.Difficulty.cheat_vision,
    'cheat_money': sc2_env.Difficulty.cheat_money,
    'cheat_insane': sc2_env.Difficulty.cheat_insane}


def get_sc2_race(sc2_race: str):
    out = sc2_races.get(sc2_race)
    if out is not None:
        return out
    else:
        raise Exception("Chosen race for StarCraft II doesn't match any known races. Try:"
                        "'terran', 'protoss', 'zerg' or 'random'")


def get_sc2_difficulty(sc2_difficulty: str):
    out = sc2_difficulties.get(sc2_difficulty)
    if out is not None:
        return out
    else:
        raise Exception("Chosen difficulty for StarCraft II doesn't match any known difficulties."
                        "Try: 'very_easy', 'easy', 'medium', 'medium_hard', 'hard',"
                        "'harder', 'very_hard', 'cheat_vision', 'cheat_money' or 'cheat_insane'")


def get_fog_of_war_percentage(obs):
    """
    This function, as the name suggests, returns the percentage of discovered fog of war
    in the match. For this, the function observes the 'minimap' features searching for
    the grid regions where its status are visible, then the counting of the total number
    of these regions is divided by the total map area. The return of this function
    is a number between [0, 1] and only the observation is needed to make it work!

    Is valid to remember about the percentage of the fog of war: the higher the worst, since
    it represents how much of the map stills unknown.
    """
    is_visible = pysc2.lib.features.Visibility.VISIBLE

    # the map area is also in the grid map (just get its size)
    map_area = obs.feature_minimap.visibility_map.flatten()
    non_visible_area = np.count_nonzero(
        map_area == (not is_visible))  # fog of war is the non-visible area
    fog_percentage = non_visible_area / map_area.size

    return fog_percentage
