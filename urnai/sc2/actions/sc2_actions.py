from pysc2.lib import actions

from urnai.actions.action_base import ActionBase

"""
This file creates dicts which store classes for each of the actions in PySC2.
"""

def _function_classes(functions):

    @classmethod
    def run_method(cls, *args) -> actions.FunctionCall:
        return cls.my_action_function(*args)
    
    function_class_dict = {}
    
    for sc2_action in functions:

        function_class_dict[sc2_action.name] = type(sc2_action.name, (ActionBase,), {

            "my_action_function": sc2_action,
            "run": run_method,

        })

    return function_class_dict

functions_classes = _function_classes(actions.FUNCTIONS)
raw_functions_classes = _function_classes(actions.RAW_FUNCTIONS)
