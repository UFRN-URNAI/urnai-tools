# File abmodel.py

## Class LearningModel

## Method __init__()

- Arguments: action_wrapper, state_builder, gamma, learning_rate, epsilon_start, epsilon_min,
  epsilon_decay_rate, per_episode_epsilon_decay, name

## Method learn()

- Arguments: s, a, r, s_, done, is_last_step

## Method choose_action()

Implements the exploration exploitation method for the model.

- Arguments: state, excluded_actions

## Method predict()

Given a State, returns the index for the action with the highest Q-Value.

- Arguments: state, excluded_actions

## Method decay_epsilon()

- No Arguments.

## Method ep_reset()

- No Arguments.
