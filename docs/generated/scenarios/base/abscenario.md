# File abscenario.py

## Class ABScenario

Abstract class for all Scenarios supported. A Scenario works like an environment, but uses it as a
base to create a trainment pattern. For example, for an environment like DeepRTS, one can train an
agent to learn how to collect specific resources, on a specific map.

A scenario also offers a default RewardBuilder and ActionWrapper to help build an agent.

## Method get_default_reward_builder()

- No Arguments.

## Method get_default_action_wrapper()

- No Arguments.
