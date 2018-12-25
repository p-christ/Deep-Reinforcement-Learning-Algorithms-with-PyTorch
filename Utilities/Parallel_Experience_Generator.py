
""" Plays n episode in parallel using a fixed agent"""
import copy
import torch
from contextlib import closing
from torch.distributions import Categorical
from multiprocessing import Pool
from Cart_Pole_Environment import Cart_Pole_Environment
from Config import Config
from Model import Model



""" ** WIP NOT FINISHED! and not working **  """



config = Config()
config.seed = 100
config.environment = Cart_Pole_Environment()
config.max_episodes_to_run = 2000
config.file_to_save_data_results = "Results_Data333.pkl"
config.file_to_save_data_results_graph = "Results_Graph333.png"
config.visualise_individual_results = True
config.visualise_overall_results = True
config.runs_per_agent = 10

config.hyperparameters = {
    "DQN_Agents": {
        "learning_rate": 0.005,
        "batch_size": 256,
        "buffer_size": 20000,
        "epsilon": 0.1,
        "discount_rate": 0.99,
        "tau": 0.1,
        "alpha_prioritised_replay": 0.6,
        "beta_prioritised_replay": 0.4,
        "incremental_td_error": 1e-8,
        "update_every_n_steps": 1,
        "nn_layers": 3,
        "nn_start_units": 20,
        "nn_unit_decay": 1.0,
        "final_layer_activation": None,
        "batch_norm": False
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
        "learning_rate": 0.001,
        "nn_layers": 2,
        "nn_start_units": 20,
        "nn_unit_decay": 1.0,
        "final_layer_activation": "SOFTMAX",
        "learning_iterations_per_round": 10,
        "discount_rate": 0.99,
        "batch_norm": False,
        "clip_epsilon": 0.1,
        "episodes_per_learning_round": 3
    }
}


def return_env():
    return Cart_Pole_Environment()


class Parallel_Experience_Generator(object):

    def __init__(self, return_env_entered, model):

        self.env =  return_env_entered
        self.model = model

    def reset_game(self):

        # env = copy.copy(self.environment)

        env = self.env()
        policy = self.model # Model(env.get_state_size(), env.get_action_size(), 2, config.hyperparameters["Policy_Gradient_Agents"]) #policy copy.deepcopy(self.policy)

        env.reset_environment()

        state = env.get_state()
        next_state = None
        action = None
        reward = None
        done = False

        episode_states = []
        episode_actions = []
        episode_rewards = []

        return env, policy, next_state, action, reward, done, state, episode_states, episode_actions, episode_rewards

    def play_n_episodes(self, env, policy, n):

        states_for_all_episodes = {}
        actions_for_all_episodes= {}
        rewards_for_all_episodes = {}


        with closing(Pool(processes=n)) as pool:
            results = pool.map(self, range(n))
            pool.terminate()

        print(len(results))
        print(len(results[0]))
        print(len(results[0][0]))

            # episode_states, episode_actions, episode_rewards = self.play_1_episode()
            #
            # states_for_all_episodes[ep] = episode_states
            # actions_for_all_episodes[ep] = episode_actions
            # rewards_for_all_episodes[ep] = episode_rewards

        return states_for_all_episodes, actions_for_all_episodes, rewards_for_all_episodes

    def play_1_episode(self):
        env, policy, next_state, action, reward, done, state, episode_states, episode_actions, episode_rewards = self.reset_game()

        while not done:
            action = self.pick_action(policy, state)

            env.conduct_action(action)

            next_state = env.get_next_state()
            reward = env.get_reward()
            done = env.get_done()

            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)

            state = next_state

        return episode_states, episode_actions, episode_rewards

    def pick_action(self, policy, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        new_policy_action_probabilities = policy.forward(state)
        action_distribution = Categorical(new_policy_action_probabilities)  # this creates a distribution to sample from
        action = action_distribution.sample().numpy()[0]
        return action


    def __call__(self, n):

        # return 5
        return self.play_1_episode()




environment = return_env()
model = Model(environment.get_state_size(), environment.get_action_size(), 2, config.hyperparameters["Policy_Gradient_Agents"]) #policy

obj = Parallel_Experience_Generator(return_env, model)
states_for_all_episodes, actions_for_all_episodes, rewards_for_all_episodes = obj.play_n_episodes(3, 3, 3)

