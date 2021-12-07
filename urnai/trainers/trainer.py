import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)

import itertools
import time
import numpy as np
import tensorflow as tf
from urnai.utils.logger import Logger
from urnai.base.savable import Savable 
from urnai.utils.reporter import Reporter as rp
from urnai.version.versioner import Versioner
from datetime import datetime

class TestParams():
    def __init__(self, num_matches, steps_per_test, max_steps=float('inf'), reward_threshold=None):
        self.num_matches = num_matches
        self.test_steps = steps_per_test
        self.max_steps = max_steps
        self.current_ep_count = 0
        self.logger = None
        self.reward_threshold = reward_threshold

class Trainer(Savable):
    ## TODO: Add an option to play every x episodes, instead of just training non-stop

    def __init__(self, env, agent, max_training_episodes, max_test_episodes, max_steps_training, max_steps_testing, save_path=os.path.expanduser("~") + os.path.sep + "urnai_saved_traingings", file_name=str(datetime.now()).replace(" ","_").replace(":","_").replace(".","_"), enable_save=True, save_every=10, relative_path=False, debug_level=0, reset_epsilon=False, tensorboard_logging=False, log_actions=True, episode_batch_avg_calculation=10, do_reward_test=False, reward_test_number_of_episodes=10, rolling_avg_window_size=20, threaded_logger_save=False):
        super().__init__()
        self.threaded_logger_save = threaded_logger_save
        self.pickle_black_list = None 
        self.prepare_black_list()
        self.setup(env, agent, max_training_episodes, max_test_episodes, max_steps_training, max_steps_testing, save_path, file_name, enable_save, save_every, relative_path, debug_level, reset_epsilon, tensorboard_logging, log_actions, episode_batch_avg_calculation=episode_batch_avg_calculation, do_reward_test=do_reward_test, reward_test_number_of_episodes=reward_test_number_of_episodes, rolling_avg_window_size=rolling_avg_window_size)

    def prepare_black_list(self):
        self.pickle_black_list = ["save_path", "file_name", "full_save_path", "full_save_play_path", "agent", "max_training_episodes","max_test_episodes","max_steps_training","max_steps_testing","save_every","rolling_avg_window_size"]

    def setup(self, env, agent, max_training_episodes, max_test_episodes, max_steps_training, max_steps_testing, save_path=os.path.expanduser("~") + os.path.sep + "urnai_saved_traingings", file_name=str(datetime.now()).replace(" ","_").replace(":","_").replace(".","_"), enable_save=True, save_every=10, relative_path=False, debug_level=0, reset_epsilon=False, tensorboard_logging=False, log_actions=True, episode_batch_avg_calculation=10, do_reward_test=False, reward_test_number_of_episodes=10, rolling_avg_window_size=20):
        self.versioner = Versioner() 
        self.env = env
        self.agent = agent
        self.save_path = save_path
        self.file_name = file_name 
        self.enable_save = enable_save
        self.save_every = save_every
        self.relative_path = relative_path
        self.reset_epsilon = reset_epsilon
        self.max_training_episodes = max_training_episodes
        self.max_test_episodes = max_test_episodes
        self.max_steps_training = max_steps_training
        self.max_steps_testing = max_steps_testing
        self.curr_training_episodes = -1
        self.curr_playing_episodes = -1
        rp.VERBOSITY_LEVEL = debug_level
        self.tensorboard_logging = tensorboard_logging
        self.log_actions = log_actions
        self.episode_batch_avg_calculation = episode_batch_avg_calculation
        self.do_reward_test=do_reward_test
        self.reward_test_number_of_episodes = reward_test_number_of_episodes
        self.rolling_avg_window_size = rolling_avg_window_size
        self.inside_training_test_loggers = []

        self.logger = Logger(0, self.agent.__class__.__name__, self.agent.model.__class__.__name__, self.agent.model, self.agent.action_wrapper.__class__.__name__, self.agent.action_wrapper.get_action_space_dim(), self.agent.action_wrapper.get_named_actions(), self.agent.state_builder.__class__.__name__, self.agent.reward_builder.__class__.__name__, self.env.__class__.__name__, log_actions=self.log_actions, episode_batch_avg_calculation=self.episode_batch_avg_calculation, rolling_avg_window_size=self.rolling_avg_window_size, threaded_saving=self.threaded_logger_save) 

        # Adding epsilon, learning rate and gamma factors to our pickle black list, 
        # so that they are not loaded when loading the model's weights.
        # Making it so that the current training session acts as a brand new training session
        # (except for the fact that the model's weights may already be somewhat optimized from previous trainings)
        if self.reset_epsilon:
            self.agent.model.pickle_black_list.append("epsilon_greedy")
            self.agent.model.pickle_black_list.append("epsilon_decay_rate")
            self.agent.model.pickle_black_list.append("epsilon_min")
            self.agent.model.pickle_black_list.append("gamma")
            self.agent.model.pickle_black_list.append("learning_rate")
            self.agent.model.pickle_black_list.append("learning_rate_min")
            self.agent.model.pickle_black_list.append("learning_rate_decay")
            self.agent.model.pickle_black_list.append("learning_rate_decay_ep_cutoff")

        currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        parentdir = os.path.dirname(currentdir)
        parentdir = os.path.dirname(parentdir)
        if(relative_path):
            self.full_save_path = parentdir + os.path.sep + self.save_path + os.path.sep + self.file_name
        else:
            self.full_save_path = self.save_path + os.path.sep + self.file_name 
        
        self.full_save_play_path = self.full_save_path + os.path.sep + "play_files"

        if self.enable_save and os.path.exists(self.full_save_path):
            rp.report("WARNING! Loading training from " + self.full_save_path + " with SAVING ENABLED.")
            self.load(self.full_save_path)
            self.versioner.ask_for_continue()
            self.make_persistance_dirs(self.log_actions)
        elif self.enable_save:
            rp.report("WARNING! Starting new training on " + self.full_save_path + " with SAVING ENABLED.")
            self.make_persistance_dirs(self.log_actions)
        else:
            rp.report("WARNING! Starting new training WITHOUT SAVING PROGRESS.")

        if(self.tensorboard_logging):
            logdir = self.full_save_path + "/tf_logs"
            self.agent.model.tensorboard_callback = [tf.keras.callbacks.TensorBoard(log_dir=logdir)]

    def make_persistance_dirs(self, log_actions):
        if log_actions:
            dir_list = [
                        self.full_save_path,
                        self.full_save_path + os.path.sep + "action_graphs" + os.path.sep + "instant",
                        self.full_save_path + os.path.sep + "action_graphs" + os.path.sep + "average",
                        self.full_save_path + os.path.sep + "action_graphs" + os.path.sep + "per_episode_bars",
                        self.full_save_path + os.path.sep + "performance_graphs",
                        self.full_save_play_path,
                        self.full_save_play_path + os.path.sep + "action_graphs" + os.path.sep + "instant",
                        self.full_save_play_path + os.path.sep + "action_graphs" + os.path.sep + "average",
                        self.full_save_play_path + os.path.sep + "action_graphs" + os.path.sep + "per_episode_bars",
                        self.full_save_play_path + os.path.sep + "performance_graphs",
                    ]
        else:
            dir_list = [
                        self.full_save_path,
                        self.full_save_path + os.path.sep + "performance_graphs",
                        self.full_save_play_path,
                        self.full_save_play_path + os.path.sep + "performance_graphs",
                    ]

        for mkdir in dir_list:
            try:
                os.makedirs(mkdir)
            except FileExistsError:
                pass

    def old_train(self, test_params: TestParams=None, reward_from_agent=True):
        start_time = time.time()

        rp.report("> Training")
        if self.logger.ep_count == 0:
            self.logger = Logger(self.max_training_episodes, self.agent.__class__.__name__, self.agent.model.__class__.__name__, self.agent.model, self.agent.action_wrapper.__class__.__name__, self.agent.action_wrapper.get_action_space_dim(), self.agent.action_wrapper.get_named_actions(), self.agent.state_builder.__class__.__name__, self.agent.reward_builder.__class__.__name__, self.env.__class__.__name__, log_actions=self.log_actions, episode_batch_avg_calculation=self.episode_batch_avg_calculation, rolling_avg_window_size=self.rolling_avg_window_size, threaded_saving=self.threaded_logger_save) 

        if test_params != None:
            test_params.logger = self.logger

        while self.curr_training_episodes < self.max_training_episodes:
            self.curr_training_episodes += 1
            self.env.start()

            # Reset the environment
            obs = self.env.reset()
            step_reward = 0
            done = False
            # Passing the episode to the agent reset, so that it can be passed to model reset
            # Allowing the model to track the episode number, and decide if it should diminish the
            # Learning Rate, depending on the currently selected strategy.
            self.agent.reset(self.curr_training_episodes)

            ep_reward = 0
            victory = False

            ep_actions = np.zeros(self.agent.action_wrapper.get_action_space_dim())
            self.logger.record_episode_start()

            for step in range(self.max_steps_training):
                
                # Choosing an action and passing it to our env.step() in order to act on our environment
                action = self.agent.step(obs, done, is_testing=False)
                obs, default_reward, done = self.env.step(action)

                is_last_step = step == self.max_steps_training - 1
                done = done or is_last_step

                # Checking whether or not to use the reward from the reward builder so we can pass that to the agent
                if reward_from_agent:
                    step_reward = self.agent.get_reward(obs, default_reward, done)
                else:
                    step_reward = default_reward

                # Making the agent learn
                self.agent.learn(obs, step_reward, done)

                # Adding our step reward to the total count of the episode's reward
                ep_reward += step_reward

                ep_actions[self.agent.previous_action] += 1

                if done:
                    victory = default_reward == 1
                    agent_info = {
                            "Learning rate" : self.agent.model.learning_rate,
                            "Gamma" : self.agent.model.gamma,
                            "Epsilon" : self.agent.model.epsilon_greedy,
                            }
                    self.logger.record_episode(ep_reward, victory, step + 1, agent_info, ep_actions)
                    break
            
            self.logger.log_ep_stats()

            #check if user wants to pause training and test agent
            #if self.do_reward_test and self.curr_training_episodes % self.episode_batch_avg_calculation == 0 and self.curr_training_episodes > 1:
            if self.do_reward_test and self.curr_training_episodes % self.episode_batch_avg_calculation == 0:
                self.test_agent()

            if self.enable_save and self.curr_training_episodes > 0 and self.curr_training_episodes % self.save_every == 0:
                self.save(self.full_save_path)

                #if we have done tests along the training
                #save all loggers for further detailed analysis
                #this was needed because the play() method
                #was saving these loggers every test, slowing down
                #training a lot. Putting this code here allows
                #to save them once and optimize training time.
                if self.do_reward_test and len(self.inside_training_test_loggers) > 0:
                    for idx in range(len(self.logger.ep_avg_batch_rewards_episodes)):
                        logger_dict = self.inside_training_test_loggers[idx]
                        if not logger_dict["saved"]:
                            episode = self.logger.ep_avg_batch_rewards_episodes[idx]
                            backup_full_save_path = self.full_save_path
                            self.full_save_path = self.full_save_path + os.path.sep + "inside_training_play_files" + os.path.sep + "test_at_training_episode_{}".format(episode)
                            self.make_persistance_dirs(self.log_actions)
                            logger_dict["logger"].save(self.full_save_path)
                            logger_dict["saved"] = True
                            self.full_save_path = backup_full_save_path



            if test_params != None and self.curr_training_episodes % test_params.test_steps == 0 and episode != 0:
                test_params.current_ep_count = self.curr_training_episodes
                self.play(test_params.num_matches, test_params.max_steps, test_params)

                # Stops training if reward threshold was reached in play testing
                if test_params.reward_threshold != None and test_params.reward_threshold <= test_params.logger.play_rewards_avg[-1]:
                    rp.report("> Reward threshold was reached!")
                    rp.report("> Stopping training")
                    break


        end_time = time.time()
        rp.report("\n> Training duration: {} seconds".format(end_time - start_time))

        self.logger.log_train_stats()
        self.logger.plot_train_stats()
        # Saving the model when the training has ended
        if self.enable_save:
            self.save(self.full_save_path)
            #if we have done tests along the training
            #save all loggers for further detailed analysis
            #this was needed because the play() method
            #was saving these loggers every test, slowing down
            #training a lot. Putting this code here allows
            #to save them once and optimize training time.
            if self.do_reward_test and len(self.inside_training_test_loggers) > 0:
                for idx in range(len(self.logger.ep_avg_batch_rewards_episodes)):
                    logger_dict = self.inside_training_test_loggers[idx]
                    if not logger_dict["saved"]:
                        episode = self.logger.ep_avg_batch_rewards_episodes[idx]
                        backup_full_save_path = self.full_save_path
                        self.full_save_path = self.full_save_path + os.path.sep + "inside_training_play_files" + os.path.sep + "test_at_training_episode_{}".format(episode)
                        self.make_persistance_dirs(self.log_actions)
                        logger_dict["logger"].save(self.full_save_path)
                        logger_dict["saved"] = True
                        self.full_save_path = backup_full_save_path

    def old_play(self, test_params=None, reward_from_agent = True):
        rp.report("\n\n> Playing")

        self.logger = Logger(self.max_test_episodes, self.agent.__class__.__name__, self.agent.model.__class__.__name__, self.agent.model, self.agent.action_wrapper.__class__.__name__, self.agent.action_wrapper.get_action_space_dim(), self.agent.action_wrapper.get_named_actions(), self.agent.state_builder.__class__.__name__, self.agent.reward_builder.__class__.__name__, self.env.__class__.__name__, log_actions=self.log_actions, episode_batch_avg_calculation=self.episode_batch_avg_calculation, rolling_avg_window_size=self.rolling_avg_window_size, threaded_saving=self.threaded_logger_save) 

        while self.curr_playing_episodes < self.max_test_episodes:
            self.curr_playing_episodes += 1
            self.env.start()

            # Reset the environment
            obs = self.env.reset()
            step_reward = 0
            done = False
            # Passing the episode to the agent reset, so that it can be passed to model reset
            # Allowing the model to track the episode number, and decide if it should diminish the
            # Learning Rate, depending on the currently selected strategy.
            self.agent.reset(self.curr_playing_episodes)

            ep_reward = 0
            victory = False

            ep_actions = np.zeros(self.agent.action_wrapper.get_action_space_dim())
            self.logger.record_episode_start()

            for step in range(self.max_steps_testing):
                action = self.agent.step(obs, done, is_testing=True)
                # Take the action (a) and observe the outcome state(s') and reward (r)
                obs, default_reward, done = self.env.step(action)

                is_last_step = step == self.max_steps_testing - 1
                done = done or is_last_step

                if reward_from_agent:
                    step_reward = self.agent.get_reward(obs, default_reward, done)
                else:
                    step_reward = default_reward

                ep_reward += step_reward

                ep_actions[self.agent.previous_action] += 1

                # If done: finish episode
                if done:
                    victory = default_reward == 1
                    agent_info = {
                            "Learning rate" : self.agent.model.learning_rate,
                            "Gamma" : self.agent.model.gamma,
                            "Epsilon" : self.agent.model.epsilon_greedy,
                            }
                    self.logger.record_episode(ep_reward, victory, step + 1, agent_info, ep_actions)
                    break

            self.logger.log_ep_stats()

        if test_params != None:
            test_params.logger.record_play_test(test_params.current_ep_count, self.logger.ep_rewards, self.logger.victories, self.max_test_episodes)
        else:
            # Only logs train stats if this is not a test, to avoid cluttering the interface with info
            self.logger.log_train_stats()

        # We need to save playing status as well 
        if self.enable_save:
            self.logger.save(self.full_save_play_path)
            rp.save(self.full_save_play_path)

    def train(self, reward_from_agent=True):
        self.training_loop(is_testing=False, reward_from_agent=reward_from_agent)

    def play(self, reward_from_agent=True):
        self.training_loop(is_testing=True, reward_from_agent=reward_from_agent)

    def training_loop(self, is_testing, reward_from_agent=True):
        start_time = time.time()
        #current_episodes = 0

        if is_testing:
            rp.report("\n\n> Playing")
            max_episodes = self.max_test_episodes
            max_steps = self.max_steps_testing
            current_episodes = self.curr_playing_episodes
        else:
            rp.report("> Training")
            max_episodes = self.max_training_episodes
            max_steps = self.max_steps_training
            current_episodes = self.curr_training_episodes
        
        if self.logger.ep_count == 0 or is_testing:
            self.logger = Logger(max_episodes, self.agent.__class__.__name__, self.agent.model.__class__.__name__, self.agent.model, self.agent.action_wrapper.__class__.__name__, self.agent.action_wrapper.get_action_space_dim(), self.agent.action_wrapper.get_named_actions(), self.agent.state_builder.__class__.__name__, self.agent.reward_builder.__class__.__name__, self.env.__class__.__name__, log_actions=self.log_actions, episode_batch_avg_calculation=self.episode_batch_avg_calculation, rolling_avg_window_size=self.rolling_avg_window_size, threaded_saving=self.threaded_logger_save) 
        

        while current_episodes < max_episodes:
            current_episodes += 1
            # starting env
            self.env.start()

            if is_testing:
                self.curr_playing_episodes = current_episodes
            else:
                self.curr_training_episodes = current_episodes

            # Reset the environment
            obs = self.env.reset()
            step_reward = 0
            done = False
            # Passing the episode to the agent reset, so that it can be passed to model reset
            # Allowing the model to track the episode number, and decide if it should diminish the
            # Learning Rate, depending on the currently selected strategy.
            self.agent.reset(current_episodes)

            ep_reward = 0
            victory = False

            ep_actions = np.zeros(self.agent.action_wrapper.get_action_space_dim())
            self.logger.record_episode_start()

            for step in range(max_steps):
                # Choosing an action and passing it to our env.step() in order to act on our environment
                action = self.agent.step(obs, done, is_testing)
                # Take the action (a) and observe the outcome state (s') and reward (r)
                obs, default_reward, done = self.env.step(action)

                # Logic to test wheter this is the last step of this episode
                is_last_step = step == max_steps - 1
                done = done or is_last_step

                # Checking whether or not to use the reward from the reward builder so we can pass that to the agent
                if reward_from_agent:
                    step_reward = self.agent.get_reward(obs, default_reward, done)
                else:
                    step_reward = default_reward

                # Making the agent learn
                if not is_testing:
                    self.agent.learn(obs, step_reward, done)

                # Adding our step reward to the total count of the episode's reward
                ep_reward += step_reward
                ep_actions[self.agent.previous_action] += 1

                if done:
                    victory = default_reward == 1
                    agent_info = {
                            "Learning rate" : self.agent.model.learning_rate,
                            "Gamma" : self.agent.model.gamma,
                            "Epsilon" : self.agent.model.epsilon_greedy,
                            }
                    self.logger.record_episode(ep_reward, victory, step + 1, agent_info, ep_actions)
                    break
            
            self.logger.log_ep_stats()

            # check if user wants to pause training and test agent
            # if self.do_reward_test and current_episodes % self.episode_batch_avg_calculation == 0 and current_episodes > 1:
            if (not is_testing) and self.do_reward_test and current_episodes % self.episode_batch_avg_calculation == 0:
                self.test_agent()

            # if this is not a test (evaluation), saving is enabled and we are in a multiple 
            # of our save_every variable then we save the model and generate graphs
            if (not is_testing) and self.enable_save and current_episodes > 0 and current_episodes % self.save_every == 0:
                self.save(self.full_save_path)

                # if we have done tests along the training save all loggers for further detailed analysis
                if self.do_reward_test and len(self.inside_training_test_loggers) > 0:
                    for idx in range(len(self.logger.ep_avg_batch_rewards_episodes)):
                        logger_dict = self.inside_training_test_loggers[idx]
                        if not logger_dict["saved"]:
                            episode = self.logger.ep_avg_batch_rewards_episodes[idx]
                            backup_full_save_path = self.full_save_path
                            self.full_save_path = self.full_save_path + os.path.sep + "inside_training_play_files" + os.path.sep + "test_at_training_episode_{}".format(episode)
                            self.make_persistance_dirs(self.log_actions)
                            logger_dict["logger"].save(self.full_save_path)
                            logger_dict["saved"] = True
                            self.full_save_path = backup_full_save_path

        end_time = time.time()
        if is_testing:
            rp.report("\n> Test duration: {} seconds".format(end_time - start_time))
            self.logger.log_train_stats()
            self.env.close()
        else:
            rp.report("\n> Training duration: {} seconds".format(end_time - start_time))
            self.logger.log_train_stats()
        
        # Saving the model at the end of the training loop
        if self.enable_save:
            if is_testing:
                self.logger.save(self.full_save_play_path)
                rp.save(self.full_save_play_path)
            else:
                self.save(self.full_save_path)

                # if we have done tests along the training save all loggers for further detailed analysis
                if self.do_reward_test and len(self.inside_training_test_loggers) > 0:
                    for idx in range(len(self.logger.ep_avg_batch_rewards_episodes)):
                        logger_dict = self.inside_training_test_loggers[idx]
                        if not logger_dict["saved"]:
                            episode = self.logger.ep_avg_batch_rewards_episodes[idx]
                            backup_full_save_path = self.full_save_path
                            self.full_save_path = self.full_save_path + os.path.sep + "inside_training_play_files" + os.path.sep + "test_at_training_episode_{}".format(episode)
                            self.make_persistance_dirs(self.log_actions)
                            logger_dict["logger"].save(self.full_save_path)
                            logger_dict["saved"] = True
                            self.full_save_path = backup_full_save_path
    
    def test_agent(self):
        #backup attributes
        max_test_episodes_backup = self.max_test_episodes
        curr_playing_episodes_backup = self.curr_playing_episodes
        logger_backup = self.logger
        #full_save_play_path_backup = self.full_save_play_path
        enable_save_backup = self.enable_save

        #set attributes to test agent
        self.enable_save = False
        #self.full_save_play_path = self.full_save_path + os.path.sep + "inside_training_play_files" + os.path.sep + "test_at_training_episode_{}".format(self.curr_training_episodes)
        #self.make_persistance_dirs(self.log_actions)
        self.max_test_episodes = self.reward_test_number_of_episodes 
        self.curr_playing_episodes = 0

        rp.report("> Starting to check current agent performance.")
        #make the agent play
        self.play()
        rp.report("> Finished checking current agent performance.")

        #get_reward_avg
        rwd_avg = self.logger.ep_avg_rewards[-1] 
        #save this logger for later saving
        #this is needed to get some more detailed
        #info on tests
        logger_dict = {}
        logger_dict["logger"] = self.logger
        logger_dict["saved"] = False
        self.inside_training_test_loggers.append(logger_dict)


        #restore backup
        self.max_test_episodes = max_test_episodes_backup
        self.curr_playing_episodes = curr_playing_episodes_backup
        self.logger = logger_backup
        #self.full_save_play_path = full_save_play_path_backup
        self.enable_save = enable_save_backup

        #register reward avg:
        self.logger.inside_training_test_avg_rwds.append(rwd_avg) 

    def save_extra(self, save_path):
        self.env.save(save_path)
        self.agent.save(save_path)
        self.logger.save(save_path)
        self.versioner.save(save_path)
        rp.save(save_path)

    def load_extra(self, save_path):
        self.agent.load(save_path)
        self.env.load(save_path)
        self.logger.load(save_path)
        self.versioner.load(save_path)
        rp.load(save_path)
