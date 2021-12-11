# File ql_table.py

## Class QLearning

## Method __init__()

- Arguments: action_wrapper, state_builder, learning_rate, gamma, epsilon, epsilon_min,
  epsilon_decay, name

## Method __set_hyperparameters()

- Arguments: learning_rate, gamma, epsilon, epsilon_min, epsilon_decay

## Method learn()

- Arguments: current_state, current_action, reward, next_state, done, is_last_step

## Method __check_state_exists()

- Arguments: state_str

## Method __update_epsilon()

- No Arguments.

## Method choose_action()

- Arguments: state, excluded_actions

## Method predict()

- Arguments: state
