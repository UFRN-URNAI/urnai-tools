# Envs

Reinforcement Learning agents learn by interacting with the environment. That's why many libraries
were invented to fullfil this purpose, such as Gym, PySC2, PyGame Learning Environment (PLE) etc.
They provide a way for an agent to perform actions on a game environment, and then return an
observation, which contains information about the environment after the action was performed on it,
a reward, which is a value that tells how good the performed action was, considering the previous
state and the current state of the environment, and done, which is a boolean value that tells
whether the game finished after the action was performed or not, which could mean a victory or a
defeat.

The thing is, all of these environments return the same information, but they don't necessarily
follow the same standards. PySC2, for example, returns all this info on a single dictionary when the
step function is called, whereas Gym returns this info from the step function in separated objects,
with an observation being a list of values of any type, a reward an integer, and done a boolean, and
PLE returns only the reward through its step function, and observation, done need to be accessed
through other methods.

Since one of our main purposes is to provide users a way to train different agents easily in a
generic environment, our library provides a generic Environment class that wraps the existing
environments and standardizes information so that all environments will return them in the same
format, which is highly based on OpenAI Gym's format. This means every time whenever we want to use
an environment that is supported by URNAI, it should be imported from the urnai.envs module.

So let's say you want to use our wrapper for the Gym environment, you should import it as follows:

```py
from urnai.envs import gym

env = gym.GymEnv(_id="gym/game/id/here")
```

## Supporting a new environment

If you want to support a new environment, you basically need to create a wrapper class for it that
extends from [urnai.envs.base.abenv.Env](./base/abenv.py).

When you extend the base Env class, you should implement all of its abstract methods, which are the
start, step, reset and close methods. Most of the times these methods are already implemented on the
environment you want to support, so you'll just have to make the new wrapper class call them from an
instance of the environment you want to support.

On the following sections we'll specify what each method of the base Env class does, how you should
implement it and show you an example of implementation using [urnai.envs.gym.GymEnv](./gym.py).

### start

This method simply starts the environment by creating a new instance of it. The only thing you'll
have to keep in mind when implementing this method in your wrapper class is that it should **set the
value of self.env_instance to an instance of the supported environment**, or it won't work.

In our Gym wrapper class, we have implemented it like this:

```py
def start(self):
    if not self.env_instance:
        self.env_instance = gym.make(self.id)
```

Note that gym.make returns an instance of the environment, so we're setting self.env_instance to an
instance of Gym.

### step

Step should receive an action from the agent as a parameter, perform this action on the instance of
the environment and return a tuple containing [Observation, Reward, Done], where:

- Observation: A list of values of any type containing information about the environment.
- Reward: An integer value that tells how good the performed action was.
- Done: A boolean value that tells whether the current match is finished or not.

As we have mentioned before, all environments already provide the user a way to access these
informations, but they are not standardized. So your goal when implementing this method will be to
execute the action passed as a parameter on the instance of the environment and return
an [Observation, Reward, Done] tuple from the environment after the action was performed.

In our Gym wrapper class, we have implemented it like this:

```py
def step(self, action):
    obs, reward, done, _ = self.env_instance.step(action)
    return obs, reward, done
```

Note that the step method from the instance of Gym also returns an extra value. To keep things
consistent, we'll discard that value and return only observation, reward and done. Since our base
Env class is highly based on Gym's Env class, this example might not be very valuable, so we'll also
show you how we did it with our [urnai.envs.ple.PLEEnv](./ple.py) environment:

```py
def step(self, action):
    reward = self.env_instance.act(action)
    obs = self.env_instance.getGameState()
    done = self.env_instance.game_over()
    return obs, reward, done
```

Notice that the first thing we do on this method is performing the action on the environment, so
that all info we return comes from the environment after the action was performed. Unlike OpenAI
Gym, the act method of the PLE instance only returns a reward, so we should get observation and done
from the getGameState and game_over methods, respectively.

### reset

Reset should simply reset the environment's instance to its initial state, which is the beginning of
the game. The only thing you should keep in mind when implementing this method for a new environment
is that it should return an observation.

In [urnai.envs.gym.GymEnv](./gym.py), we have it implemented like this:

```py
def reset(self):
    return self.env_instance.reset()
```

GymEnv's instance object already returns an observation when reset is called, but this is not always
the case, as you can observe in our [PLEEnv](./ple.py) implementation:

```py
def reset(self):
    self.env_instance.reset_game()
    obs = self.env_instance.getGameState()
    return obs
```

### close

As the name implies, close simply closes the environment by shutting down its instance. It doesn't
have to return anything and most of the environments already have a close method, so this should be
very easy to integrate.

In our GymEnv, we have implemented it as follows:

```py
def close(self):
    self.env_instance.close()
```
