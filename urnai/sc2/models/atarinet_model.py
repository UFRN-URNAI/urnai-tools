import os
import random
from collections import deque, namedtuple
from typing import Tuple

import numpy as np
import torch
from torch import nn

from urnai.models.model_base import ModelBase
from urnai.sc2.environments.sc2environment import SC2Env
from urnai.sc2.models.atarinet_neural_network import AtariNetNeuralNetwork
from urnai.sc2.models.atarinet_neural_network_simple import AtariNetNeuralNetworkSimple

Transition = namedtuple('Transition',
            ('state', 'action', 'reward', 'next_state'))

DeepmindState = namedtuple('DeepmindState',
            ('screen', 'minimap', 'nonspatial'))

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
    def __init__(self,
                 map_name,
                 replay_buffer_size = 3000//4,
                 batch_size = 64,
                 gamma = 0.99,
                 tau = 0.005,
                 learning_rate = 3e-4,
                 epsilon = 0.5,
                 n_frame_stack = 4):
        super().__init__()

        self.frame_stack = [deque(maxlen=n_frame_stack) for _ in range(3)]
        self.n_frame_stack = n_frame_stack

        self.batch_size = batch_size
        self.replay_buffer_size = replay_buffer_size
        self.gamma = gamma
        self.tau = tau
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay_value = 0.999
        self.epsilon_min = min(0.01, self.epsilon)

        self.training = True

        #self.action_spec = SC2Env(map_name=map_name).env_instance.action_spec()[0]

        self.available_actions = [i for i in range(22*16)] #TODO: link this to actionspace
        action_space_info = {"types" : [],
                              "functions" : self.available_actions}

        self.device = (torch.accelerator.current_accelerator().type 
            if torch.accelerator.is_available() else "cpu")
        print(f"\nUsing {self.device} device\n")

        #TODO: remove magic numbers (they must be sourced from the state)
        def atarinet():
            n = self.n_frame_stack
            return AtariNetNeuralNetwork(
                input_channels_screen = n * 12, #12
                input_channels_minimap = n * 2, #2
                input_channels_nonspatial = n * 11, #11
                height = 64, width = 64,
                action_space_info = action_space_info
            ).to(self.device)

        self.policy_net = atarinet()
        self.target_net = atarinet()

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True)

        self.replay_buffer = ReplayMemory(self.replay_buffer_size)

        self.total_loss = 0 #TODO: formalize

    def learn(self, state, action, reward, next_state, done) -> None:
        self.replay_buffer.push(state, action, reward, next_state)

        if len(self.replay_buffer) >= self.batch_size:
            self.optimize_model()

        self.soft_update()

    def predict(self, state) -> Tuple[int, np.ndarray]:

        if self.training and random.uniform(0, 1) < self.epsilon:
            function_id = random.choice(self.available_actions)
        else:
            with torch.no_grad():
                output = self.policy_net(self.to_DeepmindState([state]))
                function_id = output.argmax().item()
        arguments = []
        """
        for arg in self.action_spec.types:
            for dim in arguments_softmax[arg.name]:
                arg_value = np.argmax(arguments_softmax[arg.name][dim].detach().numpy())
                arguments.append(arg_value)
        """

        return function_id, arguments
    
    def optimize_model(self):
        #Original from: 
        #https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

        transitions : list[Transition] = self.replay_buffer.sample(self.batch_size)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device, dtype=torch.bool
        )

        state_batch = self.to_DeepmindState(batch.state)
        action_batch = torch.tensor(batch.action, device=self.device).unsqueeze(-1)
        reward_batch = torch.tensor(batch.reward, device=self.device)

        non_final_next_states = self.to_DeepmindState(
            [s for s in batch.next_state if s is not None]
        )

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            values = self.target_net(non_final_next_states).max(1).values
            next_state_values[non_final_mask] = values

        expected_state_action_values = next_state_values * self.gamma + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.total_loss += loss.item()

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1)
        self.optimizer.step()
        del loss, state_action_values, expected_state_action_values, next_state_values

    def soft_update(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = (policy_net_state_dict[key]*self.tau
                                    + target_net_state_dict[key]*(1-self.tau))
        self.target_net.load_state_dict(target_net_state_dict)

    def to_DeepmindState(self, batch):
        """
            Converts a batch of states from ReplayMemory into the DeepmindState format.

            Args:
                batch : list of states. Each state is a list with screen,
                    minimap and nonspatial observations respectively.
        """
        screens, minimaps, nonspatials = zip(*batch)

        # Flattening Frames and Channels into a single dimension
        n = self.n_frame_stack
        screens = [screen.reshape(n * screen.shape[1], 64, 64) for screen in screens]
        minimaps = [minimap.reshape(n * minimap.shape[1], 64, 64) for minimap in minimaps]
        nonspatials = [nonspatial.reshape(n * nonspatial.shape[1]) for nonspatial in nonspatials]

        return DeepmindState(
            torch.as_tensor(np.array(screens), device=self.device, dtype=torch.float32),
            torch.as_tensor(np.array(minimaps), device=self.device, dtype=torch.float32),
            torch.as_tensor(np.array(nonspatials), device=self.device, dtype=torch.float32),
        )
    
    def epsilon_decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay_value)

    def save(self, path):
        self.policy_net = self.policy_net.to(torch.bfloat16)
        self.target_net = self.target_net.to(torch.bfloat16)

        torch.save({
            "policy_net" : self.policy_net.state_dict(),
            "target_net" : self.target_net.state_dict(),
            #"optmizer" : self.optimizer.state_dict(),
            "epsilon" : self.epsilon,
            #"replay_buffer" : self.replay_buffer
        }, path)

        self.policy_net = self.policy_net.to(torch.float)
        self.target_net = self.target_net.to(torch.float)

    def load(self, path):
        data = torch.load(path, weights_only=False)
        self.policy_net.load_state_dict(data['policy_net'])
        self.target_net.load_state_dict(data['target_net'])

        self.policy_net = self.policy_net.to(torch.float)
        self.target_net = self.target_net.to(torch.float)

        #self.optimizer.load_state_dict(data['optmizer'])
        self.epsilon = data['epsilon']
        #self.replay_buffer = data['replay_buffer']

        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True)

    def new_ep(self):
        self.epsilon_decay()
        self.total_loss = 0
        self.clear_frame_stack()

    def make_frame_stack(self, frame):

        def make_frame_stack_single_layer(frame, layer_idx):
            if len(self.frame_stack[layer_idx]) == 0:
                for _ in range(self.n_frame_stack):
                    self.frame_stack[layer_idx].append(frame)

            self.frame_stack[layer_idx].append(frame)
        
            return np.stack(self.frame_stack[layer_idx], axis=0)

        return [make_frame_stack_single_layer(frame[layer_idx], layer_idx)
                 for layer_idx in range(len(frame))]
    
    def clear_frame_stack(self):
        for layer_idx in range(3):
            self.frame_stack[layer_idx].clear()