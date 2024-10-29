from pysc2.lib import actions
from urnai.actions.action_base import ActionBase

sc2_raw_action_classes = {}

def constructor(self):
    ...

def run_method(self, *args) -> actions.FunctionCall:
    return self.my_action_function(*args)

for sc2_action in actions.RAW_FUNCTIONS:

    sc2_raw_action_class = type(sc2_action.name, (ActionBase,), {

    "__init__" : constructor,
    "my_action_function": sc2_action,
    "run": run_method,

    })

    sc2_raw_action_classes[sc2_action.name] = sc2_raw_action_class