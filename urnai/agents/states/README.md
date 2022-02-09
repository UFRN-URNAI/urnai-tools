# State Builder

In order for a Reinforcement Learning Agent to learn it has to have some sort of representation of
the Enviroment it is interacting with. In other words, a set of information from the enviroment is
fed to the Agent's Neural Network as input so it can learn from it. The representation of the game
environment is called state representation, state observation, or simply "state".

In URNAI, the [State Builder class](./abstate.py) serves the purpuse of creating this state
representation. Through one or more StateBuilder classes we can create several "lenses" through
which the Agent can experience the environment. Each one of these "lenses" will result in different
learning, as they can completely change the way the Agent perceives its surroundings.

There are fairly simple State Builder classes, such as [PureState](./gym.py), which just uses the
default environment observation from the game environment.

However, not all State Builders can be this simple. Some game environments are so complex that the
default observation is so complex and full of information that it is unfeasible to use it raw. There
fore, the State Builder has to filter it down and only use the necessary information to make the
Agent Learn. Some examples of more complex State Builders can be seen in files such
as [VizDoom states](./vizdoom.py) and [StarCraft II states](./sc2.py)

## Creating a new State Builder

The first step towards creating a new State Builder is creating a new class and inheriting
from [StateBuilder](./abstate.py). After that, you will need to implement each base method. Below,
we give a brief overview of each method.

### build_state(self, obs)

This method receives as a parameter an Observation and returns a State, which is usually a list of
features extracted from the Observation. The Agent uses this State during training to receive a new
action from its model and also to make it learn, that's why this method should always return a list.

### def get_state_dim(self)

Returns the dimensions of the state returned by the build_state method.

OBS: The state dimension should be known before the build_state method is called. In other words,
the state size cannot be dynamically generated inside build_method, it has to be defined in
get_state_dim, because get_state_dim is called beforehand to generate the Agent's Neural Network
with the proper input size.

### def reset(self)

Resets the state builder at the end of a game episode. The reset method is called in the base reset
method inside [urnai.agents.base.abagent](../base/abagent.py)
