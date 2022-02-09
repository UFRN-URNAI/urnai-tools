# File ddqn_keras.py

## Class DDQNKeras

## Method __init__()

- Arguments: action_wrapper, state_builder, learning_rate, gamma, name, epsilon_start, epsilon_min,
  epsilon_decay, per_episode_epsilon_decay, update_target_every, batch_size, use_memory,
  memory_maxlen, min_memory_size, build_model

## Method learn()

- Arguments: s, a, r, s_, done

## Method memory_learn()

- Arguments: s, a, r, s_, done

## Method no_memory_learn()

- Arguments: s, a, r, s_, done

## Method load_extra()

- Arguments: persist_path
