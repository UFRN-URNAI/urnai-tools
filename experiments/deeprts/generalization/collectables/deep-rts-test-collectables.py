import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)

from envs.deep_rts import DeepRTSEnv
import random
import numpy as np

episodes = 100
steps = 1000 
drts = DeepRTSEnv(render=True, map='10x8-collect_twenty.json', updates_per_action = 12)

def set_collectable_list(width, height):
    map = np.zeros((height, width)) 

    for i in range(width):
        for j in range(height):
            print('i' + str(i))
            print('j' + str(j))
            map[i][j] = random.randint(0, 1)

            if np.sum(map) > 20:
                break
        else:
            continue
        break

    return map

for ep in range(episodes):
    print("Episode " + str(ep + 1))
    drts.reset()
    collectables_map = set_collectable_list(10, 8)
    epi_reward = 0

    for step in range(steps):
        print("Step " + str(step + 1))
        
        text = '''
            Choose:
                PreviousUnit = 1,
                NextUnit = 2,
                MoveLeft = 3,
                MoveRight = 4,
                MoveUp = 5,
                MoveDown = 6,
                MoveUpLeft = 7,
                MoveUpRight = 8,
                MoveDownLeft = 9,
                MoveDownRight = 10,
                Attack = 11,
                Harvest = 12,
                Build0 = 13,
                Build1 = 14,
                Build2 = 15,
                NoAction = 16
        '''

        action = None

        try:
            action = int(input(text)) - 1
        except ValueError:
            action = 15

        state, done = drts.step(action)

        unit_x = drts.players[0].get_targeted_unit().tile.x
        unit_y = drts.players[0].get_targeted_unit().tile.y

        reward = collectables_map[unit_y - 1, unit_x - 1]
        epi_reward += reward

        if (reward > 0):
            collectables_map[unit_y - 1, unit_x - 1] = 0

        print("Current state: ")
        print(state)
        print("Player 1 selected unit:")
        print(drts.players[0].get_targeted_unit())
        print("Unit coordinates: {x}, {y}".format(x=unit_x,y=unit_y))
        print("Some Player 1 stats:")
        print("Oil: {oil}".format(oil=drts.players[0].oil))
        print("Gold: {gold}".format(gold=drts.players[0].gold))
        print("Food: {food}".format(food=drts.players[0].food))
        print("Lumber: {lumber}".format(lumber=drts.players[0].lumber))
        print("Total Episode Reward: {rwd}".format(rwd=epi_reward))
        print("Step Reward: {rwd}".format(rwd=reward))
        print("Collectables map: {map}".format(map=collectables_map))


        print("FPS: " + str(drts.game.get_fps()))

        if done:
            break
