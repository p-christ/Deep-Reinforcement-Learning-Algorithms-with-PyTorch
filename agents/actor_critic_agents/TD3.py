import torch
import torch.nn.functional as functional
from torch import optim
from agents.Base_Agent import Base_Agent
from .DDPG import DDPG
from exploration_strategies.Gaussian_Exploration import Gaussian_Exploration

class TD3(DDPG):
    """A TD3 Agent from the paper Addressing Function Approximation Error in Actor-Critic Methods (Fujimoto et al. 2018)
    https://arxiv.org/abs/1802.09477"""
    agent_name = "TD3"

    def __init__(self, config):
        DDPG.__init__(self, config)
        self.critic_local_2 = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                           key_to_use="Critic", override_seed=self.config.seed + 1)
        self.critic_target_2 = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                            key_to_use="Critic")
        Base_Agent.copy_model_over(self.critic_local_2, self.critic_target_2)
        self.critic_optimizer_2 = optim.Adam(self.critic_local_2.parameters(),
                                           lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.exploration_strategy_critic = Gaussian_Exploration(self.config)

    def compute_critic_values_for_next_states(self, next_states):
        """Computes the critic values for next states to be used in the loss for the critic"""
        with torch.no_grad():
            actions_next = self.actor_target(next_states)
            actions_next_with_noise =  self.exploration_strategy_critic.perturb_action_for_exploration_purposes({"action": actions_next})
            critic_targets_next_1 = self.critic_target(torch.cat((next_states, actions_next_with_noise), 1))
            critic_targets_next_2 = self.critic_target_2(torch.cat((next_states, actions_next_with_noise), 1))
            critic_targets_next = torch.min(torch.cat((critic_targets_next_1, critic_targets_next_2),1), dim=1)[0].unsqueeze(-1)
        return critic_targets_next

    def critic_learn(self, states, actions, rewards, next_states, dones):
        """Runs a learning iteration for both the critics"""
        critic_targets_next =  self.compute_critic_values_for_next_states(next_states)
        critic_targets = self.compute_critic_values_for_current_states(rewards, critic_targets_next, dones)

        critic_expected_1 = self.critic_local(torch.cat((states, actions), 1))
        critic_expected_2 = self.critic_local_2(torch.cat((states, actions), 1))

        critic_loss_1 = functional.mse_loss(critic_expected_1, critic_targets)
        critic_loss_2 = functional.mse_loss(critic_expected_2, critic_targets)

        self.take_optimisation_step(self.critic_optimizer, self.critic_local, critic_loss_1, self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.take_optimisation_step(self.critic_optimizer_2, self.critic_local_2, critic_loss_2,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])

        self.soft_update_of_target_network(self.critic_local, self.critic_target, self.hyperparameters["Critic"]["tau"])
        self.soft_update_of_target_network(self.critic_local_2, self.critic_target_2, self.hyperparameters["Critic"]["tau"])




