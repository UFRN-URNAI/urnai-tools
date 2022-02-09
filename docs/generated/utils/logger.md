# File logger.py

## Class Logger

## Method __init__()

- Arguments: ep_total, agent_name, model_name, model_builder, action_wrapper_name,
  state_builder_name, reward_builder_name, env_name, is_episodic, render

## Method reset()

- No Arguments.

## Method record_episode()

- Arguments: ep_reward, has_won, steps_count, agent_info

## Method record_play_test()

- Arguments: ep_count, play_rewards, play_victories, num_matches

## Method log_training_start_information()

- No Arguments.

## Method log_ep_stats()

- No Arguments.

## Method log_train_stats()

- No Arguments.

## Method plot_train_stats()

- No Arguments.

## Method plot_average_reward_graph()

- No Arguments.

## Method plot_average_steps_graph()

- No Arguments.

## Method plot_instant_reward_graph()

- No Arguments.

## Method plot_win_rate_graph()

- No Arguments.

## Method plot_win_rate_percentage_over_play_testing_graph()

- No Arguments.

## Method plot_reward_average_over_play_testing_graph()

- No Arguments.

## Method save_extra()

- Arguments: persist_path

## Method __plot_curve()

- Arguments: x, y, x_label, y_label, title

## Method __plot_bar()

- Arguments: x_values, y_bars, bar_labels, x_label, y_label, title, width, format_percent,
  percent_scale

## Method __lerp()

- Arguments: a, b, t
