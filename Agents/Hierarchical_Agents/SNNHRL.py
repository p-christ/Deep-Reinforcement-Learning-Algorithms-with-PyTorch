from Base_Agent import Base_Agent


class SNNHRL(Base_Agent):
    """Implements the hierarchical RL agent that uses stochastic neural networks (SNN) from the paper Florensa et al. 2017
    https://arxiv.org/pdf/1704.03012.pdf"""
    agent_name = "SNNHRL"

    def __init__(self):
        Base_Agent.__init__(self, config)


        self.episodes_for_pretraining = 100


    def step(self):


        if self.episode_number <= self.episodes_for_pretraining:
            self.skill_training_step()

        else:
            self.post_training_step()

    def skill_training_step(self):
        """Runs a step of the game in skill training stage"""

        # pick z randomly

        # play episode with z fixed

        # calculate rewards
        # learn SNN through PPO learning

        self.pre_training()

        # then SNN learn... using PPO learning regime

    def post_training_step(self):
        """Runs a step of the game after we have frozen the SNN and are just training the manager"""

        # manager takes in state and selects skill probabilities
        # latent code sampled according to skill probabilities and fed into SNN which then acts in world for T steps
        # manager uses PPO to learn