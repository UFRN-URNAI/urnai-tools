import inspect
import os
import random
import sys

from envs.deep_rts import DeepRTSEnv
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)


def print_unit(unit):
    if (unit is not None):
        print('Unit type: {type}, unit nameID: {name_id}'.format(
            type=unit.type,
            name_id=unit.name_id,
        ))
    else:
        print('None')


def get_buildable_tile(drts_unit, drts_game):
    tile = drts_unit.tile
    x = tile.x
    y = tile.y

    map = drts_game.tilemap

    tile_list = []
    # UP_LEFT
    tile_list.append(map.get_tile(x - 1, y - 1))

    # UP
    tile_list.append(map.get_tile(x, y - 1))

    # UP_RIGHT
    tile_list.append(map.get_tile(x + 1, y - 1))

    # RIGHT
    tile_list.append(map.get_tile(x + 1, y))

    # RIGHT_DOWN
    tile_list.append(map.get_tile(x + 1, y + 1))

    # DOWN
    tile_list.append(map.get_tile(x, y + 1))

    # DOWN_RIGHT
    tile_list.append(map.get_tile(x - 1, y + 1))

    # LEFT
    tile_list.append(map.get_tile(x - 1, y))

    for tile in tile_list:
        if tile.is_buildable:
            return tile


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


def random_spawn_unit(drts_unit, drts_game, player):
    tile_map_len = len(drts_game.tilemap.tiles)
    tile = drts_game.tilemap.tiles[random.randint(0, tile_map_len - 1)]

    while not tile.is_buildable():
        tile = drts_game.tilemap.tiles[random.randint(0, tile_map_len - 1)]

    drts_game.players[player].spawn_unit(drts.constants.Unit.Archer, tile)


episodes = 100
steps = 1000
drts = DeepRTSEnv(
    map='26x14-find_and_defeat.json',
    render=True,
    updates_per_action=12,
    start_oil=99999,
    start_gold=99999,
    start_lumber=99999,
    start_food=99999,
    number_of_players=2,
)

drts.engine_config.set_footman(False)
drts.engine_config.set_archer(True)

for ep in range(episodes):
    print('Episode ' + str(ep + 1))
    drts.reset()
    collectables_map = set_collectable_list(10, 8)
    epi_reward = 0

    for step in range(steps):
        print('Step ' + str(step + 1))

        if step == 0:
            for i in range(5):
                random_spawn_unit(7, drts.game, 1)

        text = """
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
                AutoBuildArcher = 17
                BuildArcher = 18
                AutoBuildFootman = 19
        """

        action = None

        try:
            action = int(input(text)) - 1
        except ValueError:
            action = 15

        if action == 16:
            # build archer
            print('Trying to build archer')
            drts.players[0].spawn_unit_around_spawn_point(drts.constants.Unit.Archer)

            state, done = drts.step(15)
        elif action == 17:
            print('You need to input the x and y coordinates: ')
            x = int(input('X: '))
            y = int(input('Y: '))

            drts.players[0].spawn_unit(drts.constants.Unit.Archer, drts.game.tilemap.get_tile(x, y))
            state, done = drts.step(15)
        elif action == 18:
            # build archer
            print('Trying to build footman')
            drts.players[0].spawn_unit_around_spawn_point(drts.constants.Unit.Footman)

            state, done = drts.step(15)
        else:
            state, done = drts.step(action)

        unit_x = -1
        unit_y = -1
        try:
            unit_x = drts.players[0].get_targeted_unit().tile.x
            unit_y = drts.players[0].get_targeted_unit().tile.y
        except AttributeError:
            pass

        # reward = collectables_map[unit_y - 1, unit_x - 1]
        reward = 0
        epi_reward += reward

        print('Current state: ')
        print(state)
        print('Player 1 selected unit:')
        print_unit(drts.players[0].get_targeted_unit())
        print('Unit coordinates: {x}, {y}'.format(x=unit_x, y=unit_y))
        print('Some Player 1 stats:')
        print('Oil: {oil}'.format(oil=drts.players[0].oil))
        print('Gold: {gold}'.format(gold=drts.players[0].gold))
        print('Food: {food}'.format(food=drts.players[0].food))
        print('Lumber: {lumber}'.format(lumber=drts.players[0].lumber))
        print('Total Episode Reward: {rwd}'.format(rwd=epi_reward))
        print('Step Reward: {rwd}'.format(rwd=reward))
        # print('Collectables map: {map}'.format(map=collectables_map))

        print('FPS: ' + str(drts.game.get_fps()))

        if done:
            break
