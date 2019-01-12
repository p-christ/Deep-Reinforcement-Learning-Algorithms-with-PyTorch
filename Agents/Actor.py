import torch
from torch import optim
from Neural_Network import Neural_Network
from Parallel_Experience_Generator import Parallel_Experience_Generator

class Actor(object):

    def __init__(self, config, policy_input_size, policy_output_size):
        self.random_seed = config.seed
        self.hyperparameters = config.hyperparameters
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.policy = Neural_Network(policy_input_size, policy_output_size, self.random_seed,
                                         self.hyperparameters, "VANILLA_NN").to(self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.hyperparameters["learning_rate"])
        self.environment =  config.environment
        self.experience_generator = Parallel_Experience_Generator(self.environment, self.policy, self.random_seed,
                                                                  self.hyperparameters)

    def play_n_episodes(self, number_of_episodes_to_play, episode_number):
        """Plays n episodes in parallel using the fixed policy and returns the data"""
        return self.experience_generator.play_n_episodes(number_of_episodes_to_play, episode_number)

    def take_optimisation_step(self, loss):
        """Takes an optimisation step for the policy"""
        self.policy_optimizer.zero_grad() #reset gradients to 0
        loss.backward() #this calculates the gradients
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.hyperparameters["gradient_clipping_norm"]) #clip gradients to help stabilise training
        self.policy_optimizer.step() #this applies the gradients
