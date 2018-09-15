import time

from DQN_Agents.DDQN_Agent import DDQN_Agent
from DQN_Agents.DQN_Agent import DQN_Agent
from DQN_Agents.DQN_Agent_With_Fixed_Q_Targets import DQN_Agent_With_Fixed_Q_Targets
from DQN_Agents.DDQN_With_Prioritised_Experience_Replay import DDQN_With_Prioritised_Experience_Replay
from Open_AI_Gym_Environments.Cart_Pole_Environment import Cart_Pole_Environment
from Policy_Gradient_Agents.REINFORCE_Agent import REINFORCE_Agent
from Stochastic_Policy_Search_Agents.Genetic_Agent import Genetic_Agent
from Stochastic_Policy_Search_Agents.Hill_Climbing_Agent import Hill_Climbing_Agent
from Utilities import print_two_lines, produce_median_results, visualise_results_by_agent


SEED = 100
ROLLING_SCORE_LENGTH = 100
AVERAGE_SCORE_REQUIRED = 195
EPISODES_TO_RUN = 5000
FILE_TO_SAVE_DATA_RESULTS = "Episode_results_by_agent.npy"
RUNS_PER_AGENT = 10


hyperparameters = {

    "DQN_Agents": {
        "learning_rate": 0.0006,
        "batch_size": 64,
        "buffer_size": 10000,
        "epsilon": 0.1,
        "discount_rate": 0.99,
        "tau": 1e-3,
        "alpha_prioritised_replay": 0.6,
        "beta_prioritised_replay": 0.4,
        "update_every_n_steps": 1,
        "nn_layers": 3,
        "nn_start_units": 20,
        "nn_unit_decay": 1.0,
        "softmax_final_layer": False
    },

    "Stochastic_Policy_Search_Agents": {
        "policy_network_type": "Linear",
        "noise_scale_start": 1e-2,
        "noise_scale_min": 1e-3,
        "noise_scale_max": 2.0,
        "noise_scale_growth_factor": 2.0,
        "stochastic_action_decision": False,
        "num_policies": 10,
        "episodes_per_policy": 1,
        "num_policies_to_keep": 5
    },

    "Policy_Gradient_Agents": {
        "learning_rate": 0.01,
        "nn_layers": 2,
        "nn_start_units": 20,
        "nn_unit_decay": 1.0,
        "softmax_final_layer": True,
        "discount_rate": 0.99
    }
}


results = {}

agent_number = 1

agents = [Genetic_Agent, Hill_Climbing_Agent, DDQN_With_Prioritised_Experience_Replay, DQN_Agent, DQN_Agent_With_Fixed_Q_Targets, DDQN_Agent, REINFORCE_Agent]
agents = [Genetic_Agent]

ENVIRONMENT = Cart_Pole_Environment()


for agent_class in agents:

    agent_results = []

    for run in range(RUNS_PER_AGENT):

        start = time.time()

        agent_name = agent_class.__name__
        print("\033[1m" + "{}: {}".format(agent_number, agent_name) + "\033[0m", flush=True)

        agent = agent_class(ENVIRONMENT, SEED, hyperparameters,
                            ROLLING_SCORE_LENGTH, AVERAGE_SCORE_REQUIRED, agent_name)


        game_scores, rolling_scores = agent.run_n_episodes(num_episodes_to_run=EPISODES_TO_RUN, save_model=False)

        agent_number += 1
        print("Time taken: {}".format(time.time() - start), flush=True)
        print_two_lines()

        agent_results.append([game_scores, rolling_scores, len(rolling_scores), -1 * max(rolling_scores)])

    median_result = produce_median_results(agent_results)

    results[agent_name] = [median_result[0], median_result[1]]

visualise_results_by_agent(agents, results, AVERAGE_SCORE_REQUIRED)