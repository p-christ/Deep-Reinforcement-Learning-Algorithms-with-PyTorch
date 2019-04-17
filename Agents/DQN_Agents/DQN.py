import torch
import random
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from Agents.Base_Agent import Base_Agent
from Utilities.Data_Structures.Replay_Buffer import Replay_Buffer

class DQN(Base_Agent):
    """A deep Q learning agent"""
    agent_name = "DQN"
    def __init__(self, config):
        Base_Agent.__init__(self, config)
        self.memory = Replay_Buffer(self.hyperparameters["buffer_size"], self.hyperparameters["batch_size"], config.seed)
        self.q_network_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size)
        self.q_network_optimizer = optim.Adam(self.q_network_local.parameters(),
                                              lr=self.hyperparameters["learning_rate"])

    def step(self):
        """Runs a step within a game including a learning step if required"""
        while not self.done:
            self.pick_and_conduct_action()
            self.update_next_state_reward_done_and_score()
            if self.time_for_q_network_to_learn():
                self.q_network_learn()
            self.save_experience()
            self.state = self.next_state #this is to set the state for the next iteration
            self.global_step_number += 1
        self.episode_number += 1

    def pick_action(self, q_network=None, epsilon_decay_denominator=None):
        """Uses the local Q network and an epsilon greedy policy to pick an action"""
        # PyTorch only accepts mini-batches and not single observations so we have to use unsqueeze to add
        # a "fake" dimension to make it a mini-batch rather than a single observation
        if q_network is None: q_network = self.q_network_local
        state = torch.from_numpy(self.state).float().unsqueeze(0).to(self.device)
        q_network.eval() #puts network in evaluation mode
        print("STATE ", state)
        with torch.no_grad():
            action_values = q_network(state)
        q_network.train() #puts network back in training mode
        action = self.make_epsilon_greedy_choice(action_values, epsilon_decay_denominator=epsilon_decay_denominator)
        return action

    def make_epsilon_greedy_choice(self, action_values, epsilon_decay_denominator=None):
        """Chooses action with highest q_value with probability 1 - epsilon, otherwise picks randomly"""
        epsilon = self.get_updated_epsilon_exploration(epsilon_decay_denominator=epsilon_decay_denominator)
        if random.random() > epsilon: return np.argmax(action_values.data.cpu().numpy())
        return random.choice(np.arange(self.action_size))

    def q_network_learn(self, experiences=None, q_network=None, optimizer=None, replay_buffer=None, start_learning_rate=None):
        """Runs a learning iteration for the Q network"""
        if start_learning_rate is None: start_learning_rate = self.hyperparameters["learning_rate"]
        if q_network is None: q_network = self.q_network_local
        if optimizer is None: optimizer = self.q_network_optimizer
        if experiences is None: states, actions, rewards, next_states, dones = self.sample_experiences(replay_buffer=replay_buffer) #Sample experiences
        else: states, actions, rewards, next_states, dones = experiences

        loss = self.compute_loss(states, next_states, rewards, actions, dones)
        if self.done: #we only update the learning rate at end of each episode
            self.update_learning_rate(start_learning_rate, optimizer)
        self.take_optimisation_step(optimizer, q_network, loss, self.hyperparameters["gradient_clipping_norm"])

    def compute_loss(self, states, next_states, rewards, actions, dones):
        """Computes the loss required to train the Q network"""
        with torch.no_grad():
            Q_targets = self.compute_q_targets(next_states, rewards, dones)
        Q_expected = self.compute_expected_q_values(states, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        return loss

    def compute_q_targets(self, next_states, rewards, dones):
        """Computes the q_targets we will compare to predicted q values to create the loss to train the Q network"""
        Q_targets_next = self.compute_q_values_for_next_states(next_states)
        Q_targets = self.compute_q_values_for_current_states(rewards, Q_targets_next, dones)
        return Q_targets

    def compute_q_values_for_next_states(self, next_states):
        """Computes the q_values for next state we will use to create the loss to train the Q network"""
        Q_targets_next = self.q_network_local(next_states).detach().max(1)[0].unsqueeze(1)
        return Q_targets_next

    def compute_q_values_for_current_states(self, rewards, Q_targets_next, dones):
        """Computes the q_values for current state we will use to create the loss to train the Q network"""
        Q_targets_current = rewards + (self.hyperparameters["discount_rate"] * Q_targets_next * (1 - dones))
        return Q_targets_current

    def compute_expected_q_values(self, states, actions):
        """Computes the expected q_values we will use to create the loss to train the Q network"""
        Q_expected = self.q_network_local(states).gather(1, actions.long()) #must convert actions to long so can be used as index
        return Q_expected

    def locally_save_policy(self):
        """Saves the policy"""
        torch.save(self.qnetwork_local.state_dict(), "Models/{}_local_network.pt".format(self.agent_name))

    def time_for_q_network_to_learn(self):
        """Returns boolean indicating whether enough steps have been taken for learning to begin and there are
        enough experiences in the replay buffer to learn from"""
        return self.right_amount_of_steps_taken() and self.enough_experiences_to_learn_from()

    def right_amount_of_steps_taken(self):
        """Returns boolean indicating whether enough steps have been taken for learning to begin"""
        return self.global_step_number % self.hyperparameters["update_every_n_steps"] == 0

    def sample_experiences(self, replay_buffer=None):
        """Draws a random sample of experience from the memory buffer"""
        if replay_buffer is None: replay_buffer = self.memory
        experiences = replay_buffer.sample()
        states, actions, rewards, next_states, dones = experiences
        return states, actions, rewards, next_states, dones