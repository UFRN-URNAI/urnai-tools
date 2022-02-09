# File dqn_keras.py

## Class DQNKeras

## Method __init__()

- Arguments: action_wrapper, state_builder, learning_rate, gamma, name, epsilon_start, epsilon_min,
  epsilon_decay, batch_size, batch_training, memory_maxlen, use_memory, per_episode_epsilon_decay,
  build_model

## Method make_model()

- No Arguments.

## Method memorize()

- Arguments: state, action, reward, next_state, done

## Method replay()

- No Arguments.

## Method no_memory_learning()

- Arguments: s, a, r, s_, done

## Method learn()

- Arguments: s, a, r, s_, done

## Method choose_action()

- Arguments: state, excluded_actions

## Method predict()

model.predict returns an array of arrays, containing the Q-Values for the actions. This function
should return the corresponding action with the highest Q-Value.

- Arguments: state, excluded_actions

## Method save_extra()

- Arguments: persist_path

## Method load_extra()

- Arguments: persist_path
