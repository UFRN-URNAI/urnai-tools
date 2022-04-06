import numpy as np


# Both these transform methods allows us to invert the screen and minimap
# locations so that all actions can be treated as being performed from
# the top-left corner of our map.
def transformDistance(x, x_distance, y, y_distance, base_top_left):
    if not base_top_left:
        return [x - x_distance, y - y_distance]

    return [x + x_distance, y + y_distance]


def transformLocation(x, y, base_top_left, map_x=64, map_y=64):
    if not base_top_left:
        return [map_x - x, map_y - y]

    return [x, y]


def one_hot_encode(smart_actions):
    num_actions = len(smart_actions)

    encoded_actions = []
    for i in range(0, num_actions):
        encoded_action = np.zeros((num_actions,), dtype=float)
        encoded_action[i] = 1
        encoded_actions.append(encoded_action)

    return encoded_actions
