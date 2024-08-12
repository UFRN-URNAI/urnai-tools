import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)


class Trainer:
    # TODO: Add an option to play every x episodes, instead of just training non-stop

    def __init__(self, env, agent, max_training_episodes, max_playing_episodes, max_steps_training,
                 max_steps_playing,
                 ):

        self.env = env
        self.agent = agent
        self.max_training_episodes = max_training_episodes
        self.max_playing_episodes = max_playing_episodes
        self.max_steps_training = max_steps_training
        self.max_steps_playing = max_steps_playing

    def train(self, reward_from_agent=True):
        self.training_loop(is_training=True, reward_from_agent=reward_from_agent)

    def load(self, persist_path):
        self.agent.model.load(persist_path)

    def play(self, reward_from_agent=True):
        self.training_loop(is_training=False, reward_from_agent=reward_from_agent)

    def training_loop(self, is_training, reward_from_agent=True):
        current_episodes = 0

        if is_training:
            # rp.report('> Training')
            max_episodes = self.max_training_episodes
            max_steps = self.max_steps_training
        else:
            # rp.report('\n\n> Playing')
            max_episodes = self.max_playing_episodes
            max_steps = self.max_steps_playing

        while current_episodes < max_episodes:
            current_episodes += 1
            # starting env
            self.env.start()

            # Reset the environment
            obs = self.env.reset()
            step_reward = 0
            done = False
            # Passing the episode to the agent reset, so that it can be passed to model reset
            # Allowing the model to track the episode number, and decide if it should diminish the
            # Learning Rate, depending on the currently selected strategy.
            self.agent.reset(current_episodes)

            ep_reward = 0

            # ep_actions = np.zeros(self.agent.action_wrapper.get_action_space_dim())

            for step in range(max_steps):
                # Choosing an action and passing it to our env.step() in order to
                # act on our environment
                action = self.agent.step(obs[0], done, is_training)
                # Take the action (a) and observe the outcome state (s') and reward (r)
                obs, default_reward, terminated, truncated = self.env.step(action)

                done = terminated or truncated

                # Logic to test whether this is the last step of this episode
                is_last_step = step == max_steps - 1
                done = done or is_last_step

                # Checking whether or not to use the reward from the reward builder
                # so we can pass that to the agent
                if reward_from_agent:
                    step_reward = self.agent.reward.get_reward(obs[0])
                else:
                    step_reward = default_reward

                # Making the agent learn
                if is_training:
                    self.agent.learn(obs[0], step_reward, done)

                # Adding our step reward to the total count of the episode's reward
                ep_reward += step_reward
                # ep_actions[self.agent.previous_action] += 1

                if done:
                    print("Episode: %d, Reward: %d" % (current_episodes, ep_reward))
                    self.agent.model.save("saves/")
                    break

            # if this is not a test (evaluation), saving is enabled and we are in a multiple
            # of our save_every variable then we save the model and generate graphs
            # TODO
            # if is_training \
            #         and self.enable_save \
            #         and current_episodes > 0 \
            #         and current_episodes % self.save_every == 0:
            #     self.save(self.full_save_path)


        self.env.close()

        # Saving the model at the end of the training loop
        # TODO
        # if self.enable_save:
        #     if is_training:
        #         self.save(self.full_save_path)
