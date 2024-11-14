from pysc2.lib import actions

from urnai.actions.action_base import ActionBase

"""
This file creates a dict which stores a class for each of the actions in PySC2.
"""

raw_functions_classes = {}
functions_classes = {}

@classmethod
def run_method(cls, *args) -> actions.FunctionCall:
    return cls.my_action_function(*args)

for sc2_action in actions.RAW_FUNCTIONS:

    raw_function_class = type(sc2_action.name, (ActionBase,), {

        "my_action_function": sc2_action,
        "run": run_method,

    })

    raw_functions_classes[sc2_action.name] = raw_function_class

for sc2_action in actions.FUNCTIONS:

    functions_class = type(sc2_action.name, (ActionBase,), {

        "my_action_function": sc2_action,
        "run": run_method,

    })

    functions_classes[sc2_action.name] = functions_class
