from Agents.Base_Agent import Base_Agent
from Model import Model
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
        print(self.device)

        print(self.hyperparameters)

        self.memory = Replay_Buffer(self.hyperparameters["buffer_size"], self.hyperparameters["batch_size"], config.seed)
        self.critic_local = Model(self.state_size, self.action_size, config.seed, self.hyperparameters).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.hyperparameters["learning_rate"])

    def step(self):
        """Runs a step within a game including a learning step if required"""
        self.pick_and_conduct_action()
        self.update_next_state_reward_done_and_score()
        if self.time_for_critic_to_learn():
            self.critic_learn()
        self.save_experience()
        self.state = self.next_state #this is to set the state for the next iteration

    def pick_and_conduct_action(self):
        self.action = self.pick_action()
        self.conduct_action()

    def pick_action(self):
        """Uses the local Q network and an epsilon greedy policy to pick an action"""

        # PyTorch only accepts mini-batches and not single observations so we have to use unsqueeze to add
        # a "fake" dimension to make it a mini-batch rather than a single observation
        state = torch.from_numpy(self.state).float().unsqueeze(0).to(self.device)

        self.critic_local.eval() #puts network in evaluation mode
        with torch.no_grad():
            action_values = self.critic_local(state)
        self.critic_local.train() #puts network back in training mode

        action = self.make_epsilon_greedy_choice(action_values)

        return action

    def make_epsilon_greedy_choice(self, action_values):
        epsilon = self.hyperparameters["epsilon"] / (1.0 + self.episode_number / 200.0)

        if random.random() > epsilon:
            return np.argmax(action_values.data.cpu().numpy())
        return random.choice(np.arange(self.action_size))

    def critic_learn(self, experiences_given=False, experiences=None):

        if not experiences_given:
            states, actions, rewards, next_states, dones = self.sample_experiences() #Sample experiences

        else:
            states, actions, rewards, next_states, dones = experiences

        loss = self.compute_loss(states, next_states, rewards, actions, dones) #Compute the loss
        self.take_critic_optimisation_step(loss) #Take an optimisation step

    def compute_loss(self, states, next_states, rewards, actions, dones):
        Q_targets = self.compute_q_targets(next_states, rewards, dones)
        Q_expected = self.compute_expected_q_values(states, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        return loss

    def compute_q_targets(self, next_states, rewards, dones):
        Q_targets_next = self.compute_q_values_for_next_states(next_states)
        Q_targets = self.compute_q_values_for_current_states(rewards, Q_targets_next, dones)
        return Q_targets

    def compute_q_values_for_next_states(self, next_states):
        Q_targets_next = self.critic_local(next_states).detach().max(1)[0].unsqueeze(1)
        return Q_targets_next

    def compute_q_values_for_current_states(self, rewards, Q_targets_next, dones):
        Q_targets_current = rewards + (self.hyperparameters["discount_rate"] * Q_targets_next * (1 - dones))
        return Q_targets_current

    def compute_expected_q_values(self, states, actions):
        Q_expected = self.critic_local(states).gather(1, actions.long()) #must convert actions to long so can be used as index
        return Q_expected

    def take_critic_optimisation_step(self, loss):

        if self.done: #we only update the learning rate at end of each episode
            self.update_learning_rate(self.hyperparameters["learning_rate"], self.critic_optimizer)

        loss = loss

        self.critic_optimizer.zero_grad() #reset gradients to 0
        loss.backward() #this calculates the gradients
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 5) #clip gradients to help stabilise training
        self.critic_optimizer.step() #this applies the gradients

    def update_learning_rate(self, starting_lr,  optimizer):

        if len(self.rolling_results) > 0:

            last_rolling_score = self.rolling_results[-1]

            if last_rolling_score > 0.9 * self.average_score_required_to_win:
                new_lr = starting_lr / 300.0

            if last_rolling_score > 0.75 * self.average_score_required_to_win:
                new_lr = starting_lr / 100.0

            elif last_rolling_score > 0.6 * self.average_score_required_to_win:
                new_lr = starting_lr / 20.0

            elif last_rolling_score > 0.5 * self.average_score_required_to_win:
                new_lr = starting_lr / 10.0

            elif last_rolling_score > 0.25 * self.average_score_required_to_win:
                new_lr = starting_lr / 2.0

            else:
                new_lr = starting_lr

            for g in optimizer.param_groups:
                g['lr'] = new_lr

    def save_experience(self):
        self.memory.add_experience(self.state, self.action, self.reward, self.next_state, self.done)

    def locally_save_policy(self):
        pass
        # torch.save(self.qnetwork_local.state_dict(), "Models/{}_local_network.pt".format(self.agent_name))

    def time_for_critic_to_learn(self):
        return self.right_amount_of_steps_taken() and self.enough_experiences_to_learn_from()

    def right_amount_of_steps_taken(self):
        return self.episode_step_number % self.hyperparameters["update_every_n_steps"] == 0

    def enough_experiences_to_learn_from(self):
        return len(self.memory) > self.hyperparameters["batch_size"]

    def sample_experiences(self):
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences
        return states, actions, rewards, next_states, dones
