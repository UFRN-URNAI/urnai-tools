from pysc2.lib.actions import FunctionCall


class SC2Action:

    """This class encapsulates the usage of actions from pysc2"""

    def run(action_id, *args) -> FunctionCall:
        return action_id(*args)
