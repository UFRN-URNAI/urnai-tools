from DeepTDLambdaLearner import DeepTDLambdaLearner
import gym
from helper import package_state, tweak_reward

name_of_gym = 'FrozenLake-v0'
episodes = 1000

env = gym.make(name_of_gym)
n_actions = env.action_space.n
n_states = env.observation_space.n

agent = DeepTDLambdaLearner(n_actions=n_actions, n_states=n_states)

# Iterate the game
for e in range(episodes):
    state = env.reset()
    state = package_state(state)

    total_reward = 0
    done = False
    while not done:
        action, greedy = agent.get_e_greedy_action(state)
        next_state, reward, done, _ = env.step(action)
        # env.render()

        next_state = package_state(next_state)

        # Tweaking the reward to help the agent learn faster
        tweaked_reward = tweak_reward(reward, done)

        agent.learn(state, action, next_state, tweaked_reward, greedy)

        state = next_state
        total_reward += tweaked_reward

        if done:
            if reward == 1:
                print('episode: {}/{}, score: {:.2f} and goal has been found!'.format(e, episodes,
                                                                                      total_reward))
            else:
                print('episode: {}/{}, score: {:.2f}'.format(e, episodes, total_reward))
            break

    agent.reset_e_trace()
