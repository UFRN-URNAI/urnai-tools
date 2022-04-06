import os

from urnai.utils.numpy_utils import load_csv, save_iterable_as_csv, trim_matrix

x1, y1 = 22, 28
x2, y2 = 43, 43
file_path = '/Users/marcocspc/Downloads/tmp/temp_map/layer_0.csv'
map_to_trim = load_csv(file_path)
final_map = trim_matrix(map_to_trim, x1, y1, x2, y2)
save_iterable_as_csv(final_map, directory=os.path.dirname(file_path), convert_to_int=True)
