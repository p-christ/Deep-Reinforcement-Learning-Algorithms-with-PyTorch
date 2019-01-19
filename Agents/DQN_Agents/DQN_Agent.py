from Agents.Base_Agent import Base_Agent
from Utilities.Models.Neural_Network import Neural_Network
from Utilities.Data_Structures.Replay_Buffer import Replay_Buffer
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np


class DQN_Agent(Base_Agent):
    agent_name = "DQN"

    def __init__(self, config):
        Base_Agent.__init__(self, config)
        self.memory = Replay_Buffer(self.hyperparameters["buffer_size"], self.hyperparameters["batch_size"], config.seed)
        self.q_network_local = Neural_Network(self.state_size, self.action_size, config.seed, self.hyperparameters, "VANILLA_NN").to(self.device)
        self.q_network_optimizer = optim.Adam(self.q_network_local.parameters(), lr=self.hyperparameters["learning_rate"])

    def step(self):
        """Runs a step within a game including a learning step if required"""
        while not self.done:
            self.pick_and_conduct_action()
            self.update_next_state_reward_done_and_score()
            if self.time_for_q_network_to_learn():
                self.q_network_learn()
            self.save_experience()
            self.state = self.next_state #this is to set the state for the next iteration
            self.episode_step_number += 1
        self.episode_number += 1

    def pick_action(self):
        """Uses the local Q network and an epsilon greedy policy to pick an action"""

        # PyTorch only accepts mini-batches and not single observations so we have to use unsqueeze to add
        # a "fake" dimension to make it a mini-batch rather than a single observation
        state = torch.from_numpy(self.state).float().unsqueeze(0).to(self.device)

        self.q_network_local.eval() #puts network in evaluation mode
        with torch.no_grad():
            action_values = self.q_network_local(state)
        self.q_network_local.train() #puts network back in training mode

        action = self.make_epsilon_greedy_choice(action_values)
        return action

    def make_epsilon_greedy_choice(self, action_values):
        epsilon = self.hyperparameters["epsilon"] / (1.0 + (self.episode_number / self.hyperparameters["epsilon_decay_rate_denominator"]))

        if random.random() > epsilon:
            return np.argmax(action_values.data.cpu().numpy())
        return random.choice(np.arange(self.action_size))

    def q_network_learn(self, experiences_given=False, experiences=None):

        if not experiences_given:
            states, actions, rewards, next_states, dones = self.sample_experiences() #Sample experiences
        else:
            states, actions, rewards, next_states, dones = experiences

        loss = self.compute_loss(states, next_states, rewards, actions, dones)
        if self.done: #we only update the learning rate at end of each episode
            self.update_learning_rate(self.hyperparameters["learning_rate"], self.q_network_optimizer)
        self.take_optimisation_step(self.q_network_optimizer, self.q_network_local, loss, self.hyperparameters["gradient_clipping_norm"])

    def compute_loss(self, states, next_states, rewards, actions, dones):
        with torch.no_grad():
            Q_targets = self.compute_q_targets(next_states, rewards, dones)
        Q_expected = self.compute_expected_q_values(states, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        return loss

    def compute_q_targets(self, next_states, rewards, dones):
        Q_targets_next = self.compute_q_values_for_next_states(next_states)
        Q_targets = self.compute_q_values_for_current_states(rewards, Q_targets_next, dones)
        return Q_targets

    def compute_q_values_for_next_states(self, next_states):
        Q_targets_next = self.q_network_local(next_states).detach().max(1)[0].unsqueeze(1)
        return Q_targets_next

    def compute_q_values_for_current_states(self, rewards, Q_targets_next, dones):
        Q_targets_current = rewards + (self.hyperparameters["discount_rate"] * Q_targets_next * (1 - dones))
        return Q_targets_current

    def compute_expected_q_values(self, states, actions):
        Q_expected = self.q_network_local(states).gather(1, actions.long()) #must convert actions to long so can be used as index
        return Q_expected

    def locally_save_policy(self):
        pass
        # torch.save(self.qnetwork_local.state_dict(), "Models/{}_local_network.pt".format(self.agent_name))

    def time_for_q_network_to_learn(self):
        return self.right_amount_of_steps_taken() and self.enough_experiences_to_learn_from()

    def right_amount_of_steps_taken(self):
        return self.episode_step_number % self.hyperparameters["update_every_n_steps"] == 0

    def sample_experiences(self):
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences
        return states, actions, rewards, next_states, dones
