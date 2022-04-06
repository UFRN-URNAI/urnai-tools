import json
import os
from shutil import copyfile

import DeepRTS as drts
import numpy as np
from urnai.utils.error import IncorrectDeepRTSMapDataError
from urnai.utils.reporter import Reporter as rp


DRTS_MAPS_DIR = os.path.dirname(os.path.realpath(drts.python.__file__)) + '/assets/maps'
DRTS_MAP_TEMPLATE = {
    'height': 0,
    'layers': [],
    'nextobjectid': 1,
    'orientation': 'orthogonal',
    'renderorder': 'right-down',
    'tileheight': 32,
    'tilesets': [
        {
            'columns': 19,
            'firstgid': 1,
            'image': '..\/textures\/tiles.png',  # noqa: W605
            'imageheight': 659,
            'imagewidth': 626,
            'margin': 0,
            'name': 'tiles',
            'spacing': 1,
            'tilecount': 380,
            'tileheight': 32,
            'tilewidth': 32,
        }],
    'tilewidth': 32,
    'version': 1,
    'width': 0,
}


def is_map_installed(map_name):
    return os.path.exists(DRTS_MAPS_DIR + os.sep + map_name)


def install_map(map_path, force=False):
    if force or not is_map_installed(os.path.basename(map_path)):
        if not force:
            rp.report('{map} is not installed, installing on DeepRTS...'
                      .format(map=os.path.basename(map_path)))
        copyfile(map_path, DRTS_MAPS_DIR + os.sep + os.path.basename(map_path))
    else:
        rp.report('{map} is already installed.'.format(map=os.path.basename(map_path)))


def uninstall_map(map_path):
    if is_map_installed(os.path.basename(map_path)):
        os.remove(DRTS_MAPS_DIR + os.sep + os.path.basename(map_path))
        rp.report('{map} was removed.'.format(map=os.path.basename(map_path)))
    else:
        rp.report('{map} is not installed.'.format(map=os.path.basename(map_path)))


def show_available_maps():
    rp.report('Available maps on DeepRTS:')
    rp.report(os.listdir(DRTS_MAPS_DIR))


def extract_specs(map_name):
    full_path = DRTS_MAPS_DIR + os.sep + map_name
    if is_map_installed(map_name):
        with open(full_path) as json_file:
            full_map_dict = json.load(json_file)
            general_info = {}
            general_info['map_name'] = map_name
            general_info['width'] = full_map_dict['width']
            general_info['height'] = full_map_dict['height']
            with open('general_map_info.json', 'w') as outfile:
                json.dump(general_info, outfile)

            for map_layer in full_map_dict['layers']:
                layer_str = 'layer_{}'.format(full_map_dict['layers'].index(map_layer))
                map_csv = np.array(map_layer.pop('data')).reshape(-1,
                                                                  full_map_dict['width']).astype(
                    int)
                np.savetxt(layer_str + '.csv', map_csv, fmt='%i', delimiter=',')
                with open(layer_str + '.json', 'w') as outfile:
                    outfile.write(json.dumps(map_layer, indent=4))
    else:
        rp.report('{map} is not installed.'.format(map=os.path.basename(map_name)))


def build_map(specs_path):
    general_info = {}
    layers = []

    for file_name in os.listdir(specs_path):
        if 'general_map_info' in file_name:
            with open(specs_path + os.path.sep + 'general_map_info.json', 'r') as json_file:
                general_info = json.load(json_file)
        elif 'layer' in file_name:
            layer = {}
            if 'json' in file_name:
                id = int(file_name.split('_')[-1].split('.')[0])
                with open(specs_path + os.path.sep + file_name, 'r') as json_file:
                    layer = json.load(json_file)
                    csv_file_str = specs_path + os.path.sep + file_name.split('.')[0] + '.csv'
                    data = np.genfromtxt(csv_file_str, delimiter=',')
                    data = data.flatten().astype(int).tolist()
                    layer['data'] = data
                    layer['id'] = id
                    layers.append(layer)

    new_layers = layers.copy()
    for layer in layers:
        new_layers[layer.pop('id')] = layer

    final_map = DRTS_MAP_TEMPLATE.copy()
    final_map['width'] = general_info['width']
    final_map['height'] = general_info['height']
    final_map['layers'] = new_layers

    if check_map_data(final_map):
        with open(general_info['map_name'], 'w') as outfile:
            outfile.write(json.dumps(final_map, indent=4))


def check_map_data(map_dict):
    if map_dict['width'] != map_dict['height']:
        raise IncorrectDeepRTSMapDataError(
            'All DeepRTS maps are squares, this map is width {width} and height {height}.'.format(
                width=map_dict['width'], height=map_dict['height']))
    for layer in map_dict['layers']:
        if len(layer['data']) != map_dict['width'] * map_dict['height']:
            raise IncorrectDeepRTSMapDataError(
                "One of this maps' layers has a wrong size. Its length should be width"
                + ' times height long, which is not the case.')

    return True
