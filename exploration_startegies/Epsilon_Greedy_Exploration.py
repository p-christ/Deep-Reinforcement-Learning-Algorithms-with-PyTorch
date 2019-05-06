from exploration_startegies.Base_Exploration_Strategy import Base_Exploration_Strategy
import random
import torch

class Epsilon_Greedy_Exploration(Base_Exploration_Strategy):
    """Implements an epsilon greedy exploration strategy"""
    def __init__(self, config):
        super().__init__(config)

    def perturb_action_for_exploration_purposes(self, action_info):
        """Perturbs the action of the agent to encourage exploration"""
        action_values = action_info["action_values"]
        turn_off_exploration = action_info["turn_off_exploration"]
        epsilon = self.get_updated_epsilon_exploration(action_info)
        if random.random() > epsilon or turn_off_exploration:
            return torch.argmax(action_values).item()
        return random.randint(0, action_values.shape[1] - 1)

    def get_updated_epsilon_exploration(self, action_info, epsilon=1.0):
        """Gets the probability that we just pick a random action. This probability decays the more episodes we have seen"""
        episode_number = action_info["episode_number"]
        epsilon_decay_denominator = self.config.hyperparameters["epsilon_decay_rate_denominator"]
        epsilon = epsilon / (1.0 + (episode_number / epsilon_decay_denominator))
        return epsilon

    def add_exploration_rewards(self, reward_info):
        """Actions intrinsic rewards to encourage exploration"""
        return reward_info["reward"]