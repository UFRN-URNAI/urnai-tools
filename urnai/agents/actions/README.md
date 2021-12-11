# Action Wrapper

Reinforcement Learning Agents need a set of actions to interact with their environments. The Action
Wrapper class in URNAI serves the purpuse of supplying the Agent with actions to choose from, as
well as doing other processing work such as calculating which actions are not available at the
moment, checking if the current action being performed is over, and implementing new actions that
might not be available by default from the game envinronment.

The base class for all URNAI Action Wrappers is [ActionWrapper](./base/abwrapper.py). It was built
to work as easily as possible with the environment paradigm of OpenAI's Gym. Therefore, if we take a
look at our [Gym Wrapper](./gym_wrapper.py), it's easy to notice that the methods implemented are
very simple, and are basically just extensions of the underlying action objects and variables
present in Gym's Environment.

A more complex Action Wrapper can be seen in our [StarCraft II Wrapper](./sc2_wrapper.py). In this
file we have a class called SC2Wrapper, that serves as a base for our TerranWrapper class, an action
wrapper specific for the Terran SC2 race.

These wrappers are much more complex than the Gym Wrapper because they don't use the premade
environment actions from PySC2. Instead, we created actions with a higher level of abstraction that
are chosen by the AI, like "Build Supply Depot".

Such an action would have many steps if performed by a human, such as: choose a worker to do this
action, issue the correct build command and issue a queued harvest mineral command to the worker so
it doesn't sit idle after building our supply depot.

It is inside of the SC2 Action Wrappers that we call these micro actions to compose our macro "Build
Supply Depot" action. This approach simplifies learning for the Agent, making it able to learn
faster, but it also limits the ability to generalize and to act creatively, since we force it to
only use a restricted set of pre pepared actions.

## Creating a new Action Wrapper

Given the above examples of a very simple wrapper, and a more complex one, we hope that the process
of creating a new Action Wrapper feels less intimidating.

If you wish to create a new Action Wrapper for a new game environment that we haven't covered yet,
you should probably create a new class inheriting from the [base ActionWrapper](./base/abwrapper.py)
.

If you want to create an action wrapper for SC2, so you can customize your own actions and
experiments, we recommend that you inherit from either the base SC2Wrapper or the TerranWrapper,
deppending on what you wish to accomplish.

Although, at the end, the choice is always yours, and which class you inherit from is a function of
how much flexibility you want to have.
