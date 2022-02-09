# File abreward.py

## Class RewardBuilder

Every Agent needs to own an instance of this base class in order to calculate its rewards.
So every time we want to create a new agent, we should either use an existing RewardBase
implementation or create a new one.

## Method get_reward()

- Arguments: obs
