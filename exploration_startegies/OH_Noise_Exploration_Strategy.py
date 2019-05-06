from OU_Noise import OU_Noise
from exploration_startegies.Base_Exploration_Strategy import Base_Exploration_Strategy


class OH_Noise_Exploration_Strategy(Base_Exploration_Strategy):
    """Base abstract class for agent exploration strategies. Every exploration strategy must inherit from this class
    and implement the methods perturb_action_for_exploration_purposes and add_exploration_rewards"""
    def __init__(self, config):
        super().__init__(config)
        self.noise = OU_Noise(self.config.action_size, self.config.seed, self.config.hyperparameters["mu"],
                              self.config.hyperparameters["theta"], self.config.hyperparameters["sigma"])

    def perturb_action_for_exploration_purposes(self, action_info):
        """Perturbs the action of the agent to encourage exploration"""
        action = action_info["action"]
        action += self.noise.sample()
        return action

    def add_exploration_rewards(self, reward_info):
        """Actions intrinsic rewards to encourage exploration"""
        self.noise.reset()