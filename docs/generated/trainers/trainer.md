# File trainer.py

## Class TestParams

### Method __init__()

- Arguments: num_matches, steps_per_test, max_steps, reward_threshold

## Class Trainer

### Method __init__()

- Arguments: env, agent, save_path, file_name, enable_save, save_every, relative_path, debug_level

### Method setup()

- Arguments: env, agent, save_path, file_name, enable_save, save_every, relative_path, debug_level

### Method train()

- Arguments: num_episodes, max_steps, test_params, reward_from_agent

### Method play()

- Arguments: num_matches, max_steps, test_params, reward_from_agent

### Method save_extra()

- Arguments: save_path

### Method load_extra()

- Arguments: save_path
