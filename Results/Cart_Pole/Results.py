import time

from DQN_Agents.DDQN_Agent import DDQN_Agent
from DQN_Agents.DQN_Agent import DQN_Agent
from DQN_Agents.DQN_Agent_With_Fixed_Q_Targets import DQN_Agent_With_Fixed_Q_Targets
from DQN_Agents.DDQN_With_Prioritised_Experience_Replay import DDQN_With_Prioritised_Experience_Replay
from Open_AI_Gym_Environments.Cart_Pole_Environment import Cart_Pole_Environment
from Stochastic_Policy_Search_Agents.Hill_Climbing_Agent import Hill_Climbing_Agent
from Utilities import print_two_lines, save_score_results, visualise_results_by_agent
import numpy as np

SEED = 100
ROLLING_SCORE_LENGTH = 100
AVERAGE_SCORE_REQUIRED = 195
EPISODES_TO_RUN = 1000
FILE_TO_SAVE_DATA_RESULTS = "Episode_results_by_agent.npy"

hyperparameters = {
    "learning_rate": 0.0006,
    "batch_size": 64,
    "buffer_size": 2000,
    "fc_units": [20, 10],
    "epsilon": 0.5,
    "gamma":  0.99,
    "tau": 1e-3,
    "update_every_n_steps": 1,
    "policy_network_type": "Linear"
    # "alpha": 0.5,
    # "incremental_priority": 1e-5
}

results = {}


agent_number = 1

agents = [Hill_Climbing_Agent, DQN_Agent, DQN_Agent_With_Fixed_Q_Targets, DDQN_Agent, DDQN_With_Prioritised_Experience_Replay]
#
ENVIRONMENT = Cart_Pole_Environment()


for agent_class in agents:

    start = time.time()

    agent_name = agent_class.__name__
    print("\033[1m" + "{}: {}".format(agent_number, agent_name) + "\033[0m", flush=True)

    agent = agent_class(ENVIRONMENT, SEED, hyperparameters,
                        ROLLING_SCORE_LENGTH, AVERAGE_SCORE_REQUIRED, agent_name)
    game_scores, rolling_scores = agent.run_game_n_times(num_episodes_to_run=EPISODES_TO_RUN, save_model=False)
    results[agent_name] = [game_scores, rolling_scores]
    agent_number += 1
    print("Time taken: {}".format(time.time() - start), flush=True)
    print_two_lines()

visualise_results_by_agent(agents, results, AVERAGE_SCORE_REQUIRED)