import copy
import torch
from torch import optim
from Base_Agent import Base_Agent
from Neural_Network import Neural_Network
from Replay_Buffer import Replay_Buffer
import torch.nn.functional as functional
from Utilities.OU_Noise import OU_Noise

# TODO currently critic takes state and action choice in at layer 1 but it should concatonate them later in the network

class DDPG_Agent(Base_Agent):
    agent_name = "DDPG"

    def __init__(self, config):
        Base_Agent.__init__(self, config)
        self.hyperparameters = config.hyperparameters
        self.critic_local = Neural_Network(self.state_size + self.action_size, 1, self.random_seed,
                                           self.hyperparameters["Critic"], "VANILLA_NN").to(self.device)
        self.critic_target = copy.deepcopy(self.critic_local).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=self.hyperparameters["Critic"]["learning_rate"])
        self.memory = Replay_Buffer(self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"],
                                    self.random_seed)
        self.actor_local = Neural_Network(self.state_size, self.action_size, self.random_seed,
                                          self.hyperparameters["Actor"], "VANILLA_NN").to(self.device)
        self.actor_target = copy.deepcopy(self.actor_local).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),
                                          lr=self.hyperparameters["Actor"]["learning_rate"])
        self.noise = OU_Noise(self.action_size, self.random_seed, self.hyperparameters["mu"],
                              self.hyperparameters["theta"], self.hyperparameters["sigma"])

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        Base_Agent.reset_game(self)
        self.noise.reset()

    def step(self):
        """Runs a step in the game"""
        while not self.done:
            self.pick_and_conduct_action()
            self.update_next_state_reward_done_and_score()
            if self.time_for_critic_and_actor_to_learn():
                for _ in range(self.hyperparameters["learning_updates_per_learning_session"]):
                    states, actions, rewards, next_states, dones = self.memory.sample()  # Sample experiences
                    self.critic_learn(states, actions, rewards, next_states, dones)
                    self.actor_learn(states)
            self.save_experience()
            self.state = self.next_state #this is to set the state for the next iteration
            self.episode_step_number += 1
        self.episode_number += 1

    def pick_action(self):
        """Picks an action using the actor network and then adds some noise to it to ensure exploration"""
        state = torch.from_numpy(self.state).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        action += self.noise.sample()
        return action

    def critic_learn(self, states, actions, rewards, next_states, dones):
        loss = self.compute_loss(states, next_states, rewards, actions, dones)
        self.take_optimisation_step(self.critic_optimizer, self.critic_local, loss, self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.critic_local, self.critic_target, self.hyperparameters["Critic"]["tau"])

    def compute_loss(self, states, next_states, rewards, actions, dones):
        with torch.no_grad():
            critic_targets = self.compute_critic_targets(next_states, rewards, dones)
        critic_expected = self.compute_expected_critic_values(states, actions)
        loss = functional.mse_loss(critic_expected, critic_targets)
        return loss

    def compute_critic_targets(self, next_states, rewards, dones):
        critic_targets_next = self.compute_critic_values_for_next_states(next_states)
        critic_targets = self.compute_critic_values_for_current_states(rewards, critic_targets_next, dones)
        return critic_targets

    def compute_critic_values_for_next_states(self, next_states):
        with torch.no_grad():
            actions_next = self.actor_target(next_states)
            critic_targets_next = self.critic_target(torch.cat((next_states, actions_next), 1))
        return critic_targets_next

    def compute_critic_values_for_current_states(self, rewards, critic_targets_next, dones):
        critic_targets_current = rewards + (self.hyperparameters["discount_rate"] * critic_targets_next * (1 - dones))
        return critic_targets_current

    def compute_expected_critic_values(self, states, actions):
        critic_expected = self.critic_local(torch.cat((states, actions), 1))
        return critic_expected

    def time_for_critic_and_actor_to_learn(self):
        return self.enough_experiences_to_learn_from() and self.episode_step_number % self.hyperparameters["update_every_n_steps"] == 0

    def actor_learn(self, states):
        if self.done: #we only update the learning rate at end of each episode
            self.update_learning_rate(self.hyperparameters["Actor"]["learning_rate"], self.actor_optimizer)
        actor_loss = self.calculate_actor_loss(states)
        self.take_optimisation_step(self.actor_optimizer, self.actor_local, actor_loss,
                                    self.hyperparameters["Actor"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.actor_local, self.actor_target, self.hyperparameters["Actor"]["tau"])

    def calculate_actor_loss(self, states):
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(torch.cat((states, actions_pred), 1)).mean()
        return actor_loss