from os.path import expanduser, sep
import time

import numpy as np


def load_csv(file_path, delimiter=','):
    return np.genfromtxt(file_path, delimiter=delimiter)


def save_iterable_as_csv(iterable, file_name='urnai_iterable_' + time.strftime('%Y%m%d_%H%M%S'),
                         directory=expanduser('~'), convert_to_int=False, convert_to_string=False,
                         delimiter=','):
    """
    Saves any iterable supported by numpy (such as List)
    as a csv file. It converts all values into int by default.
    """
    if '.csv' not in file_name:
        file_name += '.csv'

    csv = None
    if convert_to_int:
        csv = np.array(iterable).astype(int)
        np.savetxt(directory + sep + file_name, csv, fmt='%i', delimiter=delimiter)
    else:
        csv = np.array(iterable)
        np.savetxt(directory + sep + file_name, csv, delimiter=delimiter)


def trim_matrix(matrix, x1, y1, x2, y2):
    """
    If you have a 2D numpy array
    and you want a submatrix of that array,
    you can use this function to extract it.
    You just need to tell this function
    what are the top-left and bottom-right
    corners of this submatrix, by setting
    x1, y1 and x2, y2.
    For example: some maps of StarCraft II
    have parts that are not walkable, this
    happens specially in PySC2 mini-games
    where only a small portion of the map
    is walkable. So, you may want to trim
    this big map (generally a 64x64 matrix)
    and leave only the useful parts.
    """
    matrix = np.delete(matrix, np.s_[0:x1:1], 1)
    matrix = np.delete(matrix, np.s_[0:y1:1], 0)
    matrix = np.delete(matrix, np.s_[x2 - x1 + 1::1], 1)
    matrix = np.delete(matrix, np.s_[y2 - y1 + 1::1], 0)
    return matrix
