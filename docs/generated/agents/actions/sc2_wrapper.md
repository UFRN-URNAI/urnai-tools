# File sc2_wrapper.py

## Class SC2Wrapper

### Method __init__()

- No Arguments.

### Method is_action_done()

- No Arguments.

### Method reset()

- No Arguments.

### Method get_actions()

- No Arguments.

### Method split_action()

Breaks out (x, y) coordinates from a named action, if there are any.

- Arguments: smart_action

### Method get_excluded_actions()

- Arguments: obs

### Method get_action()

- Arguments: action_idx, obs

## Class TerranWrapper

### Method __init__()

- No Arguments.

### Method get_excluded_actions()

- Arguments: obs

### Method get_action()

- Arguments: action_idx, obs

## Class SimpleTerranWrapper

### Method __init__()

- No Arguments.

### Method get_excluded_actions()

- Arguments: obs

## Class ProtossWrapper

### Method __init__()

- No Arguments.

### Method get_action()

- Arguments: action_idx, obs

## Class ZergWrapper

### Method __init__()

- No Arguments.

### Method get_excluded_actions()

- Arguments: obs

### Method get_action()

- Arguments: action_idx, obs
