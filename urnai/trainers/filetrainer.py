from urnai.utils.module_specialist import get_cls
from urnai.utils.error import ClassNotFoundError, FileFormatNotSupportedError
from urnai.utils.file_util import is_json_file, is_csv_file 
from .trainer import Trainer
import inspect
import json, csv
import os
import pandas as pd

class FileTrainer(Trainer):

    def __init__(self, file_path):
        #this is needed because in python3
        #attributes cannot be declared outside init
        #so here we go
        self.env = None
        self.agent = None
        self.save_path = None
        self.file_name = None
        self.enable_save = None
        self.save_every = None
        self.relative_path = None
        self.reset_epsilon = None
        self.max_training_episodes = None
        self.max_test_episodes = None
        self.max_steps_training = None
        self.max_steps_testing = None
        self.curr_training_episodes = None 
        self.curr_playing_episodes = None
        self.episode_batch_avg_calculation = None
        self.do_reward_test = None
        self.reward_test_number_of_episodes = None
        self.inside_training_test_loggers = None
        self.rolling_avg_window_size = None

        self.pickle_black_list = None
        self.prepare_black_list()
        self.trainings = []
        if is_json_file(file_path):
            self.load_json_file(file_path)
        elif is_csv_file(file_path):
            self.load_csv_file(file_path)
        else:
            raise FileFormatNotSupportedError("FileTrainer only supports JSON and CSV formats.")

    def start_training(self, play_only=False, setup_only=False):
        scenario = False
        self.check_trainings()
        for training in self.trainings:
            try:
                env_cls = get_cls("urnai.envs", training["env"]["class"])
                self.remove_nonused_class_attrs(env_cls, training["env"]["params"])
                env = env_cls(**training["env"]["params"]) 
            except ClassNotFoundError as cnfe:
                if "was not found in urnai.envs" in str(cnfe):
                    env_cls = get_cls("urnai.scenarios", training["env"]["class"])
                    self.remove_nonused_class_attrs(env_cls, training["env"]["params"])
                    env = env_cls(**training["env"]["params"])
                    scenario = True

            if not scenario:
                action_wrapper_cls = get_cls("urnai.agents.actions", training["action_wrapper"]["class"])
                self.remove_nonused_class_attrs(action_wrapper_cls, training["action_wrapper"]["params"])
                action_wrapper = action_wrapper_cls(**training["action_wrapper"]["params"])

                state_builder_cls = get_cls("urnai.agents.states", training["state_builder"]["class"])
                self.remove_nonused_class_attrs(state_builder_cls, training["state_builder"]["params"])
                state_builder = state_builder_cls(**training["state_builder"]["params"])

                reward_cls = get_cls("urnai.agents.rewards", training["reward"]["class"])
                self.remove_nonused_class_attrs(reward_cls, training["reward"]["params"])
                reward = reward_cls(**training["reward"]["params"])
            else:
                action_wrapper = env.get_default_action_wrapper() 
                state_builder = env.get_default_state_builder() 
                reward = env.get_default_reward_builder() 

            model_cls = get_cls("urnai.models", training["model"]["class"])
            self.remove_nonused_class_attrs(model_cls, training["model"]["params"])
            model = model_cls(action_wrapper=action_wrapper, state_builder=state_builder, **training["model"]["params"])

            agent_cls = get_cls("urnai.agents", training["agent"]["class"])
            self.remove_nonused_class_attrs(agent_cls, training["agent"]["params"])
            agent = agent_cls(model, reward, **training["agent"]["params"])

            self.setup(env, agent, **training["trainer"]["params"])

            if not setup_only:
                if not play_only:
                    self.train()

                self.play()

    def check_trainings(self):
        for training in self.trainings:
            #sometimes when loading from csv, params are not present
            #this code fixes it
            for key in training:
                if 'params' not in training[key].keys(): 
                    training[key]['params'] = {}

            #when loading from csv, model_builder is transformed
            #into a string, this fixes it
            if 'build_model' in training['model']['params'].keys():
                if isinstance(training['model']['params']['build_model'], str):
                    string = training['model']['params']['build_model']
                    string = string.replace("'", "\"")
                    string = string.replace("None", "null")
                    training['model']['params']['build_model'] = json.loads(string)

    def load_json_file(self, json_file_path):
        with open(json_file_path, "r") as json_file:
            self.trainings = json.loads(json_file.read())

    def load_csv_file(self, csv_file_path):
        df = pd.read_csv(csv_file_path)  
        self.trainings = self.df_to_formatted_json(df)

    def save_trainings_as_csv(self, path):
        df = pd.json_normalize(self.trainings) 
        df.to_csv(path, index=False)

    def save_trainings_as_json(self, path):
        with open(path, "w+") as out_file:
            out_file.write(json.dumps(self.trainings, indent=4))

    def save_extra(self, save_path):
        super().save_extra(save_path)

        json_path = save_path + os.path.sep + "training_params.json"
        csv_path = json_path.replace('.json', '.csv') 
        self.save_trainings_as_json(json_path)
        self.save_trainings_as_csv(csv_path)

    def df_to_formatted_json(self, df, sep="."):
        """
        The opposite of json_normalize
        """
        result = []
        for idx, row in df.iterrows():
            parsed_row = {}
            for col_label,v in row.items():
                keys = col_label.split(".")

                current = parsed_row
                for i, k in enumerate(keys):
                    if i==len(keys)-1:
                        current[k] = v
                    else:
                        if k not in current.keys():
                            current[k] = {}
                        current = current[k]
            # save
            result.append(parsed_row)
        return result

    def remove_nonused_class_attrs(self, py_class, training_dict):
        class_attributes = inspect.signature(py_class).parameters

        for param in list(training_dict):
            if param not in class_attributes:
                training_dict.pop(param)
