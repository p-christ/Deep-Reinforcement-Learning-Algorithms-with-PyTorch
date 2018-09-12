import random
import numpy as np

from Base_Agent import Base_Agent


class REINFORCE_Agent(Base_Agent):

    def __init__(self, environment, seed, hyperparameters, rolling_score_length, average_score_required,
                 agent_name):
        Base_Agent.__init__(self, environment=environment,
                            seed=seed, hyperparameters=hyperparameters, rolling_score_length=rolling_score_length,
                            average_score_required=average_score_required, agent_name=agent_name)
