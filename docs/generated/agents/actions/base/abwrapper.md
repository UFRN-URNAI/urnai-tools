# File abwrapper.py

## Class ActionWrapper

ActionWrapper works as an extra abstraction layer used by the agent to select actions.
This means the agent doesn't select actions from action_set, but from ActionWrapper.
This class is responsible to telling the agents which actions it can use and which ones
are excluded from selection. It can also force the agent to use certain actions by
combining them into multiple steps

## Method __init__()

- No Arguments.

## Method is_action_done()

- No Arguments.

## Method reset()

- No Arguments.

## Method get_actions()

- No Arguments.

## Method get_excluded_actions()

- Arguments: obs

## Method get_action()

- Arguments: action_idx, obs

## Method get_action_space_dim()

- No Arguments.
