from statistics import mean

from pysc2.env import sc2_env

from urnai.sc2.actions import sc2_actions_aux as scaux
from urnai.sc2.actions.sc2_actions import raw_functions_classes as sc2_actions

def move_direction(obs, direction : dict):
    army = scaux.select_army(obs, sc2_env.Race.terran)
    xs = [unit.x for unit in army]
    ys = [unit.y for unit in army]

    new_army_x = int(mean(xs)) + direction["x"]
    new_army_y = int(mean(ys)) + direction["y"]

    return_actions = []

    for unit in army:
        return_actions.append(
            sc2_actions["Move_pt"].run(
                'now', unit.tag,[new_army_x, new_army_y]))
        
    return return_actions

def move_left(obs, hor_threshold = 2):
    return move_direction(obs, {"x" : -hor_threshold, "y" : 0})

def move_right(obs, hor_threshold = 2):
    return move_direction(obs, {"x" : +hor_threshold, "y" : 0})

def move_down(obs, ver_threshold = 2):
    return move_direction(obs, {"x" : 0, "y" : +ver_threshold})

def move_up(obs, ver_threshold = 2):
    return move_direction(obs, {"x" : 0, "y" : -ver_threshold})