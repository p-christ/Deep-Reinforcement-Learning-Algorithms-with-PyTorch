import torch
from torch.distributions.normal import Normal
from Base_Agent import Base_Agent
from OU_Noise import OU_Noise
from Replay_Buffer import Replay_Buffer
from TD3 import TD3
import copy

class SAC(Base_Agent):
    """Soft Actor-Critic model based on the Open AI implementation explained here https://spinningup.openai.com/en/latest/algorithms/sac.html"""

    # Newer version: Soft Actor-Critic Algorithms and Applications
    # https: // towardsdatascience.com / soft - actor - critic - demystified - b8427df61665

    def __init__(self, config):
        Base_Agent.__init__(self, config)

        self.hyperparameters = config.hyperparameters
        self.critic_local = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1, key_to_use="Critic")
        self.critic_target = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1, key_to_use="Critic")
        Base_Agent.copy_model_over(self.critic_local, self.critic_target)

        self.critic_local_2 = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                           key_to_use="Critic", override_seed=self.config.seed + 1)
        self.critic_target_2 = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                            key_to_use="Critic")
        Base_Agent.copy_model_over(self.critic_local_2, self.critic_target_2)

        self.critic_optimizer_2 = torch.optim.Adam(self.critic_local_2.parameters(),
                                                   lr=self.hyperparameters["Critic"]["learning_rate"])

        self.action_noise_std = self.hyperparameters["action_noise_std"]
        self.action_noise_distribution = Normal(torch.Tensor([0.0]), torch.Tensor([self.action_noise_std]))
        self.action_noise_clipping_range = self.hyperparameters["action_noise_clipping_range"]

        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(),
                                           lr=self.hyperparameters["Critic"]["learning_rate"])
        self.memory = Replay_Buffer(self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"],
                                    self.config.seed)
        self.actor_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size * 2, key_to_use="Actor")
        self.actor_target = self.create_NN(input_dim=self.state_size, output_dim=self.action_size * 2, key_to_use="Actor")
        Base_Agent.copy_model_over(self.actor_local, self.actor_target)

        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(),
                                          lr=self.hyperparameters["Actor"]["learning_rate"])
        self.noise = OU_Noise(self.action_size, self.config.seed, self.hyperparameters["mu"],
                              self.hyperparameters["theta"], self.hyperparameters["sigma"])

        self.value_function_local = self.create_NN(input_dim=self.state_size, output_dim=1,
                                           key_to_use="Value")
        self.value_function_target = self.create_NN(input_dim=self.state_size, output_dim=1,
                                            key_to_use="Value")
        self.valu_function_optimizer = torch.optim.Adam(self.value_function_local.parameters(),
                                                 lr=self.hyperparameters["Critic"]["learning_rate"])


        Base_Agent.copy_model_over(self.value_function_local, self.value_function_target)

    def critic_learn(self, states, actions, rewards, next_states, dones):

        with torch.no_grad():
            v_targets = rewards + self.hyperparameters["discount_rate"] * (1 - dones) * self.value_function_target(next_states)







