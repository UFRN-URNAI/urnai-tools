import inspect
import os
from statistics import mean
import sys

from absl import flags
from agents.actions import sc2 as scaux
from envs.sc2 import SC2Env
from pysc2.env import sc2_env
from pysc2.lib import actions

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

HOR_THRESHOLD = 2
VER_THRESHOLD = 2
PENDING_ACTIONS = []


def print_army_mean(obs):
    army = scaux.select_army(obs, sc2_env.Race.terran)
    xs = [unit.x for unit in army]
    ys = [unit.y for unit in army]

    print('Army position is [{x}, {y}]'.format(x=int(mean(xs)), y=int(mean(ys))))


def move_left(obs):
    army = scaux.select_army(obs, sc2_env.Race.terran)
    xs = [unit.x for unit in army]
    ys = [unit.y for unit in army]

    new_army_x = int(mean(xs)) - HOR_THRESHOLD
    new_army_y = int(mean(ys))

    for unit in army:
        PENDING_ACTIONS.append(
            actions.RAW_FUNCTIONS.Move_pt('now', unit.tag, [new_army_x, new_army_y]))


def move_right(obs):
    army = scaux.select_army(obs, sc2_env.Race.terran)
    xs = [unit.x for unit in army]
    ys = [unit.y for unit in army]

    new_army_x = int(mean(xs)) + HOR_THRESHOLD
    new_army_y = int(mean(ys))

    for unit in army:
        PENDING_ACTIONS.append(
            actions.RAW_FUNCTIONS.Move_pt('now', unit.tag, [new_army_x, new_army_y]))


def move_down(obs):
    army = scaux.select_army(obs, sc2_env.Race.terran)
    xs = [unit.x for unit in army]
    ys = [unit.y for unit in army]

    new_army_x = int(mean(xs))
    new_army_y = int(mean(ys)) + VER_THRESHOLD

    for unit in army:
        PENDING_ACTIONS.append(
            actions.RAW_FUNCTIONS.Move_pt('now', unit.tag, [new_army_x, new_army_y]))


def move_up(obs):
    army = scaux.select_army(obs, sc2_env.Race.terran)
    xs = [unit.x for unit in army]
    ys = [unit.y for unit in army]

    new_army_x = int(mean(xs))
    new_army_y = int(mean(ys)) - VER_THRESHOLD

    for unit in army:
        PENDING_ACTIONS.append(
            actions.RAW_FUNCTIONS.Move_pt('now', unit.tag, [new_army_x, new_army_y]))


def no_op():
    PENDING_ACTIONS.append(actions.RAW_FUNCTIONS.no_op())


def main():
    players = [sc2_env.Agent(sc2_env.Race.terran)]
    scii = SC2Env(map_name='CollectMineralShards', render=True, step_mul=32, players=players)
    episodes = 100
    steps = 99999999999999

    for ep in range(episodes):
        print('Episode ' + str(ep + 1))
        scii.reset()
        state = None

        for step in range(steps):
            print('Step ' + str(step + 1))

            action = None

            if state is not None and len(PENDING_ACTIONS) == 0:
                # ask for direction
                string = """ Choose:
                1 - Up
                2 - Down
                3 - Left
                4 - Right
                5 - No-op
    """

                # need to insert a way to choose actions
                try:
                    action = int(input(string))
                except ValueError:
                    action = 5
                print('pending action list empty')

                # check if it is not at the limit of screen
                # move it to desired direction
                if action == 1:
                    move_up(state)
                elif action == 2:
                    move_down(state)
                elif action == 3:
                    move_left(state)
                elif action == 4:
                    move_right(state)

            state, reward, done = None, None, None
            if len(PENDING_ACTIONS) > 0:
                state, reward, done = scii.step([PENDING_ACTIONS.pop()])
            else:
                state, reward, done = scii.step([actions.RAW_FUNCTIONS.no_op()])

            print('Reward: {r}'.format(r=reward))
            print_army_mean(state)

            if done:
                break


FLAGS = flags.FLAGS
FLAGS(sys.argv)
main()
