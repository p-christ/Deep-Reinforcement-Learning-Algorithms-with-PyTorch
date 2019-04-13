import copy
import torch
import torch.nn.functional as functional
from torch.distributions.normal import Normal
from torch import optim
from DDPG import DDPG

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
        self.critic_target_2.load_state_dict(copy.deepcopy(self.critic_local_2.state_dict()))
        self.critic_optimizer_2 = optim.Adam(self.critic_local_2.parameters(),
                                           lr=self.hyperparameters["Critic"]["learning_rate"])

        self.action_noise_std = self.hyperparameters["action_noise_std"]
        self.action_noise_distribution = Normal(torch.Tensor([0.0]), torch.Tensor([self.action_noise_std]))
        self.action_noise_clipping_range = self.hyperparameters["action_noise_clipping_range"]

    def compute_critic_values_for_next_states(self, next_states):
        """Computes the critic values for next states to be used in the loss for the critic"""
        with torch.no_grad():
            actions_next = self.actor_target(next_states)

            action_noise = self.action_noise_distribution.sample(sample_shape=actions_next.shape)
            action_noise = action_noise.squeeze(-1)
            clipped_action_noise = torch.clamp(action_noise, min=-self.action_noise_clipping_range,
                                               max = self.action_noise_clipping_range)
            actions_next_with_noise = actions_next + clipped_action_noise

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




