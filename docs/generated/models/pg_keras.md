# File pg_keras.py

## Class PGKeras

## Method __init__()

- Arguments: action_wrapper, state_builder, learning_rate, gamma, name, build_model

## Method make_model()

- No Arguments.

## Method custom_loss()

- Arguments: y_true, y_pred

## Method memorize()

- Arguments: state, action, reward

## Method learn()

- Arguments: s, a, r, s_, done

## Method compute_discounted_R()

- Arguments: R

## Method choose_action()

Choose action for policy gradient is equal as predict. Since there is no explore probability, all
actions will come from the Net's weights.

- Arguments: state, excluded_actions

## Method save_extra()

- Arguments: persist_path

## Method load_extra()

- Arguments: persist_path

## Method predict()

model.predict returns an array of arrays, containing the Q-Values for the actions. This function
uses the action probabilities from our policy to randomly select an action.

- Arguments: state, excluded_actions

## Method ep_reset()

- No Arguments.
