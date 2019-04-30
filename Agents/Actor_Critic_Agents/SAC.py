import torch
import torch.nn.functional as functional
from torch.distributions.normal import Normal
from Base_Agent import Base_Agent
from OU_Noise import OU_Noise
from Replay_Buffer import Replay_Buffer
from TD3 import TD3
from Utilities.Data_Structures.Tanh_Distribution import TanhNormal

import numpy as np
import copy

class SAC(Base_Agent):
    """Soft Actor-Critic model based on the Open AI implementation explained here https://spinningup.openai.com/en/latest/algorithms/sac.html"""

# Based on https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/sac/sac.py
    # Newer version: Soft Actor-Critic Algorithms and Applications
    # https: // towardsdatascience.com / soft - actor - critic - demystified - b8427df61665

    agent_name = "SAC"

    def __init__(self, config):
        Base_Agent.__init__(self, config)
        self.hyperparameters = config.hyperparameters
        self.critic_local = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1, key_to_use="Critic")
        self.critic_local_2 = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                           key_to_use="Critic", override_seed=self.config.seed + 1)
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(),
                                                 lr=self.hyperparameters["Critic"]["learning_rate"])
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_local_2.parameters(),
                                                   lr=self.hyperparameters["Critic"]["learning_rate"])
        self.critic_target = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                           key_to_use="Critic")
        self.critic_target_2 = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                            key_to_use="Critic")
        Base_Agent.copy_model_over(self.critic_local, self.critic_target)
        Base_Agent.copy_model_over(self.critic_local_2, self.critic_target_2)
        self.memory = Replay_Buffer(self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"],
                                    self.config.seed)

        self.actor_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size * 2, key_to_use="Actor")
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(),
                                          lr=self.hyperparameters["Actor"]["learning_rate"])

        self.target_entropy = -np.prod(self.environment.action_space.shape).item()  # heuristic value from Tuomas
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.hyperparameters["Actor"]["learning_rate"])

    def step(self):
        """Runs a step in the game"""
        while not self.done:
            # print("State ", self.state.shape)
            self.action, _, _, _, _, _ = self.pick_action()
            self.conduct_action(self.action)
            if self.time_for_critic_and_actor_to_learn():
                for _ in range(self.hyperparameters["learning_updates_per_learning_session"]):
                    states, actions, rewards, next_states, dones = self.sample_experiences()
                    new_actions, log_action_prob = self.temperature_learn(states)
                    actor_loss = self.calculate_actor_loss(states, new_actions, log_action_prob)
                    critic_loss_1, critic_loss_2 = self.calculate_critic_losses(states, actions, next_states, rewards, dones)

                    self.take_optimisation_step(self.critic_optimizer, self.critic_local, critic_loss_1,
                                                self.hyperparameters["Critic"]["gradient_clipping_norm"])
                    self.take_optimisation_step(self.critic_optimizer_2, self.critic_local_2, critic_loss_2,
                                                self.hyperparameters["Critic"]["gradient_clipping_norm"])
                    self.take_optimisation_step(self.actor_optimizer, self.actor_local, actor_loss,
                                                self.hyperparameters["Actor"]["gradient_clipping_norm"])
            self.save_experience()
            self.state = self.next_state #this is to set the state for the next iteration
            self.global_step_number += 1
        self.episode_number += 1

    def temperature_learn(self, states):

        action, action_mean, log_std, log_prob, _, _ = self.pick_action(state=states, track_grads=True)

        alpha_loss = -(self.alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return action, log_prob

    def calculate_actor_loss(self, states, new_actions, log_action_prob):



        q_new_actions = torch.min(self.critic_local(torch.cat((states, new_actions), 1)), self.critic_local_2(torch.cat((states, new_actions), 1)))
        policy_loss = (self.alpha * log_action_prob - q_new_actions).mean()
        return policy_loss

    def calculate_critic_losses(self, states, actions, next_states, rewards, dones):

        q1_pred = self.critic_local(torch.cat((states, actions), 1))
        q2_pred = self.critic_local_2(torch.cat((states, actions), 1))

        new_next_actions, _, _, new_log_prob, _, _ = self.pick_action(state=next_states, track_grads=True)



        critic_targets_next = torch.min(self.critic_target(torch.cat((next_states, new_next_actions), 1)),
                                    self.critic_target_2(torch.cat((next_states, new_next_actions), 1))) - self.alpha * new_log_prob
        q_target = self.compute_critic_values_for_current_states(rewards, critic_targets_next, dones)

        qf1_loss = functional.mse_loss(q1_pred, q_target.detach())
        qf2_loss = functional.mse_loss(q2_pred, q_target.detach())
        return qf1_loss, qf2_loss

    def time_for_critic_and_actor_to_learn(self):
        """Returns boolean indicating whether there are enough experiences to learn from and it is time to learn for the
        actor and critic"""
        return self.enough_experiences_to_learn_from() and self.global_step_number % self.hyperparameters["update_every_n_steps"] == 0

    def calculate_action_info(self, state=None):
        if state is None: state = torch.from_numpy(self.state).float().unsqueeze(0).to(self.device)
        actor_output = self.actor_local(state).data
        action_mean = actor_output[:, :self.action_size]
        log_std = actor_output[:, self.action_size:]
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        tanh_normal = TanhNormal(action_mean, std)
        action, pre_tanh_value = tanh_normal.rsample(return_pretanh_value=True)
        log_prob = tanh_normal.log_prob(action, pre_tanh_value=pre_tanh_value)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, action_mean, log_std, log_prob, std, pre_tanh_value

    def pick_action(self, state=None, track_grads=False):
        """Picks an action using the actor network and then adds some noise to it to ensure exploration"""
        if not track_grads:
            with torch.no_grad():
                action, action_mean, log_std, log_prob, std, pre_tanh_value = self.calculate_action_info(state=state)
        else:
            action, action_mean, log_std, log_prob, std, pre_tanh_value = self.calculate_action_info(state=state)
        return action, action_mean, log_std, log_prob, std, pre_tanh_value




    def compute_critic_values_for_current_states(self, rewards, critic_targets_next, dones):
        """Computes the critic values for current states to be used in the loss for the critic"""
        critic_targets_current = rewards + (self.hyperparameters["discount_rate"] * critic_targets_next * (1.0 - dones))
        return critic_targets_current

    def sample_experiences(self):
        return  self.memory.sample()

    @property
    def alpha(self):
        return self.log_alpha.exp()


    # def value_function_learn(self, states):
    #
    #     new_actions, log_action_prob = self.pick_action(states,  give_log_prob=True, track_grads=True)
    #
    #     # print(new_actions.requires_grad)
    #
    #     with torch.no_grad():
    #         critic_expected_1 = self.critic_local(torch.cat((states, new_actions), 1))
    #         critic_expected_2 = self.critic_local_2(torch.cat((states, new_actions), 1))
    #         critic_expected_min = torch.min(critic_expected_1, critic_expected_2)
    #
    #     target_value_func = critic_expected_min - self.hyperparameters["entropy_loss_weight"] * log_action_prob
    #
    #     value_func_values = self.value_function_local(states)
    #
    #     value_loss = functional.mse_loss(value_func_values, target_value_func)
    #
    #     self.take_optimisation_step(self.value_function_optimizer, self.value_function_local, value_loss,
    #                                 self.hyperparameters["Value"]["gradient_clipping_norm"], retain_graph=True)
    #     self.soft_update_of_target_network(self.value_function_local, self.value_function_target, self.hyperparameters["Value"]["tau"])
    #
    #     return new_actions, log_action_prob
    #

