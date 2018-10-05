import os
import sys

from Stochastic_Policy_Search_Agents.Hill_Climbing_Agent import Hill_Climbing_Agent

print("Adding project home to path")
nb_dir = os.path.split(os.getcwd())[0]
PROJECT_PATH = os.path.split(nb_dir)[0]

if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

from Environments.Unity_Environments.Banana_Environment import Banana_Environment
from Agents.DQN_Agents.DQN_Agent import DQN_Agent
from Agents.DQN_Agents.DQN_Agent_With_Fixed_Q_Targets import DQN_Agent_With_Fixed_Q_Targets
from Agents.DQN_Agents.DDQN_Agent import DDQN_Agent
from Utilities.Utilities import print_two_empty_lines, save_score_results

ENVIRONMENT = None

SEED = 100
ROLLING_SCORE_LENGTH = 100
AVERAGE_SCORE_REQUIRED = 13
EPISODES_TO_RUN = 1000
FILE_TO_SAVE_DATA_RESULTS = "Episode_results_by_agent.npy"
UNITY_FILE_NAME = "Banana.app"

if ENVIRONMENT is None:
    ENVIRONMENT = Banana_Environment(UNITY_FILE_NAME)

hyperparameters = {
    "learning_rate": 5e-4,
    "batch_size": 64,
    "buffer_size": int(1e5),
    "fc_units": [30, 30],
    "epsilon": 0.05,
    "gamma": 0.99,
    "tau": 1e-3,
    "update_every_n_steps": 1,
    "alpha": 0.5,
    "incremental_priority": 1e-5,
    "policy_network_type": "Linear"
}

results = {}

# DDQN_Agent_With_Fixed_Q_Targets_And_Prioritised_Experience_Replay

agent_number = 1

agents = [Hill_Climbing_Agent, DQN_Agent, DQN_Agent_With_Fixed_Q_Targets, DDQN_Agent]

for agent_class in agents:
    agent_name = agent_class.__name__
    print("\033[1m" + "{}: {}".format(agent_number, agent_name) + "\033[0m")

    agent = agent_class(ENVIRONMENT, SEED, hyperparameters,
                        ROLLING_SCORE_LENGTH, AVERAGE_SCORE_REQUIRED, agent_name)
    game_scores, rolling_scores = agent.run_n_episodes(num_episodes_to_run=EPISODES_TO_RUN)
    results[agent_name] = [game_scores, rolling_scores]
    agent_number += 1
    print_two_empty_lines()

save_score_results(FILE_TO_SAVE_DATA_RESULTS, results)
