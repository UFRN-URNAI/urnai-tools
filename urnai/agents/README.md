# Agents

The main purpose of a Deep Reinforcement Learning agent is to learn by interacting with the
environment. This is accomplished through a reward function, which tells our agent whether an action
he took in a certain state was good or not. An agent's success is determined by how well it can
learn its reward function and by how good it is at solving the examined problem.

The shape of a reward function is also heavily influenced by two things: The agent's representation
of the environment, also known as the State, and the actions it can perform. In URNAI-Tools, we
allow the user to create its own reward function, state representations, and to pick which actions
he wants his agent to perform. These things can be accomplished through the use
of [RewardBuilders](./rewards), [StateBuilders](./states) and [ActionWrappers](./actions),
respectively.

But even if your rewards, states and actions are good enough, some problems can only be solved by
using the correct algorithm. That's why we've also allowed users to give agents any DRL model they
want without having to make any changes to states, rewards, actions or the model itself. In
URNAI-Tools all of these elements are very modular and only need to be plugged to the agent to work.

To summarize, one of the main advantages of using URNAI-Tools to develop a DRL agent is that it's
very easy to change an agent's reward function, state representation, list of possible actions and
DRL model. This allows the user to iterate very fast and discover what works and what doesn't.

## Using an existing agent

First of all, before using an agent, you have to make sure that the environment you're going to use
is supported by URNAI, so check the [env](../envs) section. If the environment is supported, there
is also an agent class for it, so all you have to do is either implement
a [RewardBuilder](./rewards/abreward.py), [StateBuilder](./states/abstate.py)
, [ActionWrapper](./actions/base/abwrapper.py) and [DRL model](../models), or use any of the
existing ones. You'll probably want to implement a RewardBuilder and StateBuilder, since those are
highly problem dependant and the default ones don't provide good enough info to your agent, but the
default ActionWrapper for the environment you've chosen is probably good enough for solving it, and
unless you want to use a model that URNAI-Tools doesn't support, it's better to use the ones we
provide, since they've already been tested and can be parameterized to your needs.

After choosing the StateBuilder, RewardBuilder, ActionWrapper and DRL model you're going to use, all
you have to do is instantiate them and pass them to the agent's constructor.

The following example is from our agent that solves Gym's Cartpole-v1:

```py
from agents.gym_agent import GymAgent
from agents.actions.gym_wrapper import GymWrapper
from agents.rewards.default import PureReward
from agents.states.gym import PureState
from models.dqn_keras import DQNKeras

# Instantiating the State Builder
state_builder = PureState(env)

# Instantiating an ActionWrapper
action_wrapper = GymWrapper(env)

# Instantiating a Deep Q-Learning model. The StateBuilder and ActionWrapper are used as
# parameters of the model, since they control the model's input and output dimensions

dq_network = DQNKeras(action_wrapper, 
                      state_builder,
                      gamma=0.99, learning_rate=0.001, 
                      epsilon_decay=0.9997, epsilon_min=0.01)

# Instantiating a RewardBuilder
reward_builder = PureReward()

# Initializing our Gym cartpole agent
agent = GymAgent(dq_network, reward_builder)
```

## Implementing a new agent

You'll usually want to implement a new agent when you're supporting a new environment. All you have
to do is create a new class for your agent and extend
from [urnai.agents.base.abagent.Agent](./base/abagent.py) and implement its abstract methods. You'll
notice that some methods like build_state, get_reward, get_state_dim, reset and learn are already
implemented in the base class, that's because these methods are assumed to work equally for every
agent. In the sections below we'll explain what each method from the base class does and what you
should pay attention to when implementing/overriding them.

### \_\_init__(self, model: LearningModel, reward_builder: RewardBase)

#### Parameters

- model: The DRL model used by the agent. The model contains references to the StateBuilder and
  ActionWrapper used by the agent. This must be an instance of
  the [urnai.models.base.abmodel.LearningModel](../models/base/abmodel.py) class.
- reward_builder: The agent's reward function. This parameter must be an instance of
  the [urnai.agents.rewards.abreward.RewardBase](./rewards/abreward.py) class

#### What it does

The constructor method from the base class is responsible for setting many important variables for
the agent, including the StateBuilder and ActionWrapper, which are references obtained from the
model, so it should always be called from the derived class' constructor.

### build_state(self, obs)

#### Parameters

- obs: An observation from the environment. For more info, check the [env](../envs) section.

#### What it does

This method is already implemented in the base class, and it simply returns the build_state method
from the agent's StateBuilder.

### get_reward(self, obs, reward, done)

#### Parameters

- obs: An observation from the environment. For more info, check the [env](../envs) section.
- reward: The reward given by the environment when the agent performs an action. **Don't confuse it
  with the agent's reward**.
- done: Boolean value returned from the environment that tells whether the current match/episode has
  finished or not.

#### What it does

This method is already implemented in the base class, and it simply returns the get_reward method
from the agent's RewardBuilder.

### get_state_dim(self)

#### What it does

This method is already implemented in the base class, and it simply returns the get_state_dim method
from the agent's StateBuilder.

### reset(self)

#### What it does

This method is already implemented in the base class. It's called in the start of a new episode or
match, and it's responsible for resetting some of the agent's variables, as well as the
ActionWrapper. This method should be overidden whenever you want to reset something else at the
start of an episode/match, but it's important that the derived class still calls this method from
the base class.

### learn(self, obs, reward, done)

#### Parameters

- obs: An observation from the environment. For more info, check the [env](../envs) section.
- reward: The reward given by the environment when the agent performs an action. **Don't confuse it
  with the agent's reward**.
- done: Boolean value returned from the environment that tells whether the current match/episode has
  finished or not.

#### What it does

This method is already implemented in the base class. It's called in every episode step right after
an action is performed on the environment, and it's responsible for calling the learn method from
the agent's model.

#### Keep in mind

The step method receives obs, reward and done from the environment **before** the action is
performed, and this method receives these parameters **after** the action is performed. Train works
by using a (state, action, reward, next state) tuple, so in this scenario, the fact that step
receives an observation from before the action is performed means it builds state and returns
action, and the fact that learn receives an observation from the environment after the action was
performed means it can build next state and calculate the reward. This is what guarantees that the
training works.

### step(self, obs, reward, done)

#### Parameters

- obs: An observation from the environment. For more info, check the [env](../envs) section.
- reward: The reward given by the environment when the agent performs an action. **Don't confuse it
  with the agent's reward**.
- done: Boolean value returned from the environment that tells whether the current match/episode has
  finished or not.

#### What it does

This is an abstract method that must be implemented by your agent. It's responsible for getting an
action index from the agent's model and returning it. To do so, this method should build the current
state of the environment, which is used by the model to choose an action.

#### Keep in mind

The learn method receives obs, reward and done from the environment **after** the action is
performed, and this method receives these parameters **before** the action is performed. Train works
by using a (state, action, reward, next state) tuple, so in this scenario, the fact that step
receives an observation from before the action is performed means it builds state and returns
action, and the fact that learn receives an observation from the environment after the action was
performed means it can build next state and calculate the reward. This is what guarantees that the
training works.

It's also important to note that this method should set the agent's self.previous_action parameter
to the action index that it will return, and self.previous_state to the state used to choose this
action.

### play(self, obs)

#### Parameters

- obs: An observation from the environment. For more info, check the [env](../envs) section.

#### What it does

This is an abstract method that must be implemented by your agent. This works just like step, but
instead of simply getting an action index from the model, it should make the model predict an action
index, instead of using an exploration/exploitation method. More details on
the [urnai.models](../models).

#### Keep in mind

This method should set the agent's self.previous_action parameter to the action index it's going to
return.
