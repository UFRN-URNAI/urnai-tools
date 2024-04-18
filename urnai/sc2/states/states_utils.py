from pysc2.lib import features


def get_my_units_amount(obs, unit_type):
    return len(get_units_by_type(obs, unit_type, features.PlayerRelative.SELF))

def get_units_by_type(obs, unit_type, alliance=features.PlayerRelative.SELF):
    return [unit for unit in obs.raw_units
            if unit.unit_type == unit_type
            and unit.alliance == alliance
            and unit.build_progress == 100]