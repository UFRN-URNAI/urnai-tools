import random
from collections import deque, namedtuple

import numpy as np
import torch
from torch import nn

from urnai.models.model_base import ModelBase
from urnai.sc2.environments.sc2environment import SC2Env
from urnai.sc2.models.atarinet_neural_network import AtariNetNeuralNetwork

Transition = namedtuple('Transition',
            ('state', 'action', 'reward', 'next_state'))

class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class AtariNetModel(ModelBase):
    def __init__(self, map_name):
        super().__init__()

        self.action_spec = SC2Env(map_name=map_name).env_instance.action_spec()[0]

        available_actions = ["no_op", "Move_screen"]
        action_space_info = {"types" : self.action_spec.types,
                              "functions" : available_actions}

        self.device = (torch.accelerator.current_accelerator().type 
            if torch.accelerator.is_available() else "cpu")
        print(f"Using {self.device} device")

        #TODO: remove magic numbers (they must be sourced from the state)
        def atarinet():
            return AtariNetNeuralNetwork(
                input_channels_screen = 19,
                input_channels_minimap = 9,
                input_channels_nonspatial = 11,
                height = 64, width = 64,
                action_space_info = action_space_info
            ).to(self.device)

        self.policy_net = atarinet()
        self.target_net = atarinet()

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.batch_size = 128
        self.gamma = 0.99
        self.tau = 0.005
        self.learning_rate = 3e-4

        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True)

        self.replay_buffer = ReplayMemory(10)

    def learn(self, state, action, reward, next_state, done) -> None:
        self.replay_buffer.push(state, action, reward, next_state)

        if self.replay_buffer.size() >= self.batch_size:
            self.optimize_model()

        self.soft_update()

    def predict(self, state) -> int: #TODO: change the type

        inputs_screen, inputs_minimap, inputs_nonspatial = self.process_input(state)

        function_id_softmax = self.policy_net(
            (inputs_screen, inputs_minimap, inputs_nonspatial)
        )

        function_id = np.argmax(function_id_softmax.detach().numpy())

        arguments = []
        """
        for arg in self.action_spec.types:
            for dim in arguments_softmax[arg.name]:
                arg_value = np.argmax(arguments_softmax[arg.name][dim].detach().numpy())
                arguments.append(arg_value)
        """

        return function_id, arguments
    
    def optimize_model(self):
        #Sourced from: 
        #https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

        transitions = self.replay_buffer.sample(self.batch_size)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device, dtype=torch.bool
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(
                                            non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def soft_update(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = (policy_net_state_dict[key]*self.tau
                                    + target_net_state_dict[key]*(1-self.tau))
        self.target_net.load_state_dict(target_net_state_dict)

    def process_input(self, state):

        def to_input(state):
            return (torch.from_numpy(state)).unsqueeze(0)

        inputs_screen = to_input(state["screen"])
        inputs_minimap = to_input(state["minimap"])
        inputs_nonspatial = to_input(state["non_spatial"])

        return inputs_screen, inputs_minimap, inputs_nonspatial

