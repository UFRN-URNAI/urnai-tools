from absl import app
from pysc2.env import sc2_env
from urnai.envs.sc2 import SC2Env
from pysc2.lib import actions, features, units
from urnai.agents.actions import sc2 as scaux
from urnai.utils.error import NoEnemyArmyError
from statistics import mean
import math

HOR_THRESHOLD = 2
VER_THRESHOLD = 2
MAXIMUM_ATTACK_RANGE = 9

global PENDING_ACTIONS
PENDING_ACTIONS = []
#USING THRESHOLDS = 2
#Map walkable size 24 (H) x 12 (V)


def print_army_mean(obs):
    army = scaux.select_army(obs, sc2_env.Race.terran)
    xs = [unit.x for unit in army]
    ys = [unit.y for unit in army]

    print("Army position is [{x}, {y}]".format(x=int(mean(xs)),y=int(mean(ys))))

def move_left(obs):
    army = scaux.select_army(obs, sc2_env.Race.terran)
    xs = [unit.x for unit in army]
    ys = [unit.y for unit in army]

    new_army_x = int(mean(xs)) - HOR_THRESHOLD
    new_army_y = int(mean(ys))

    for unit in army:
        PENDING_ACTIONS.append(actions.RAW_FUNCTIONS.Move_pt("now", unit.tag, [new_army_x, new_army_y]))
        
def move_right(obs):
    army = scaux.select_army(obs, sc2_env.Race.terran)
    xs = [unit.x for unit in army]
    ys = [unit.y for unit in army]

    new_army_x = int(mean(xs)) + HOR_THRESHOLD
    new_army_y = int(mean(ys))

    for unit in army:
        PENDING_ACTIONS.append(actions.RAW_FUNCTIONS.Move_pt("now", unit.tag, [new_army_x, new_army_y]))

def move_down(obs):
    army = scaux.select_army(obs, sc2_env.Race.terran)
    xs = [unit.x for unit in army]
    ys = [unit.y for unit in army]

    new_army_x = int(mean(xs))
    new_army_y = int(mean(ys)) + VER_THRESHOLD

    for unit in army:
        PENDING_ACTIONS.append(actions.RAW_FUNCTIONS.Move_pt("now", unit.tag, [new_army_x, new_army_y]))

def move_up(obs):
    army = scaux.select_army(obs, sc2_env.Race.terran)
    xs = [unit.x for unit in army]
    ys = [unit.y for unit in army]

    new_army_x = int(mean(xs))
    new_army_y = int(mean(ys)) - VER_THRESHOLD

    for unit in army:
        PENDING_ACTIONS.append(actions.RAW_FUNCTIONS.Move_pt("now", unit.tag, [new_army_x, new_army_y]))

def no_op():
    PENDING_ACTIONS.append(actions.RAW_FUNCTIONS.no_op())

def get_nearest_enemy_unit_inside_radius(x, y, obs, radius):
    enemy_army = [unit for unit in obs.raw_units if unit.owner != 1] 

    closest_dist = 9999999999999 
    closest_unit = None
    for unit in enemy_army:
        xaux = unit.x
        yaux = unit.y

        dist = abs(math.hypot(x - xaux, y - yaux))
        print("dist "+str(dist))

        if dist <= closest_dist and dist <= radius:
            closest_dist = dist
            closest_unit = unit

    if closest_unit is not None:
        return closest_unit

def attack_nearest_inside_radius(obs, radius):
    #get army coordinates
    army = scaux.select_army(obs, sc2_env.Race.terran)
    xs = [unit.x for unit in army]
    ys = [unit.y for unit in army]
    army_x = int(mean(xs))
    army_y = int(mean(ys)) - VER_THRESHOLD

    #get nearest unit
    enemy_unit = get_nearest_enemy_unit_inside_radius(army_x, army_y, obs, radius)

    #tell each unit in army to attack nearest enemy
    for unit in army:
        PENDING_ACTIONS.append(actions.RAW_FUNCTIONS.Attack_pt("now", unit.tag, [enemy_unit.x, enemy_unit.y]))


def set_collectable_list(width, height):
    map = np.zeros((height, width)) 

    for i in range(width):
        for j in range(height):
            map[i][j] = random.randint(0, 1)

            if np.sum(map) > 20:
                break
        else:
            continue
        break

    return map

def main():
    episodes = 100
    steps = 1000 
    players = [sc2_env.Agent(sc2_env.Race.terran)]
    scii = SC2Env(map_name="FindAndDefeatZerglings", render=True, step_mul=48, players=players)

    for ep in range(episodes):
        print("Episode " + str(ep + 1))
        scii.reset()
        state = None
        episodes = 100
        steps = 99999999999999 

        for step in range(steps):
            print("Step " + str(step + 1))
            
            text = '''
                Choose:
                    1 - Up
                    2 - Down
                    3 - Left
                    4 - Right
                    5 - Attack Nearest Unit 
                    6 - No-Op
            '''

            action = None

            try:
                action = int(input(text))
            except ValueError:
                action = 6

            #move it to desired direction
            if state is not None:
                if action == 1:
                    move_up(state)
                elif action == 2:
                    move_down(state)
                elif action == 3:
                    move_left(state)
                elif action == 4:
                    move_right(state)
                elif action == 5:
                    try:
                        attack_nearest_inside_radius(state, MAXIMUM_ATTACK_RANGE)
                    except NoEnemyArmyError:
                        print("No enemies inside defined attack range.")

            if len(PENDING_ACTIONS) > 0:
                state, reward, done = scii.step([PENDING_ACTIONS.pop()])
            else:
                state, reward, done = scii.step([actions.RAW_FUNCTIONS.no_op()])

            #Reward is 1 for every roach killed 
            #Reward is not cumulative
            print("Reward: {r}".format(r=reward))


            #print("Current state: ")
            #print(state)

            #print its coordinates
            if state is not None:
                print_army_mean(state)

            if done:
                break

            if done:
                break

import sys
from absl import flags
FLAGS = flags.FLAGS
FLAGS(sys.argv)
main()
