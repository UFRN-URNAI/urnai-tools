# File abenv.py

## Class Env

Abstract Base Class for all environments currently supported.
Environments are classes used to create a link between agents, models and
the game. For cases where an environment for a game already exists, this class
should still be used as a wrapper (e.g. implementing an environment for OpenAI gym).

## Method __init__()

- Arguments: _id, render, reset_done

## Method start()

- No Arguments.

## Method step()

- Arguments: action

## Method reset()

- No Arguments.

## Method close()

- No Arguments.
