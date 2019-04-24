"""Tests for the hierarchical RL agent HIRO"""
import copy

import gym
import random
import numpy as np
import torch

from Hierarchical_Agents.HIRO import HIRO
from Utilities.Data_Structures.Config import Config

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

config = Config()
config.seed = 1
config.environment = gym.make("Pendulum-v0")
config.num_episodes_to_run = 1500
config.file_to_save_data_results = None
config.file_to_save_results_graph = None
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 1
config.use_GPU = False
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = False




config.hyperparameters = {

        "LOWER_LEVEL": {
            "max_lower_level_timesteps": 3,

            "Actor": {
                "learning_rate": 0.001,
                "linear_hidden_units": [20, 20],
                "final_layer_activation": "TANH",
                "batch_norm": False,
                "tau": 0.005,
                "gradient_clipping_norm": 5
            },

            "Critic": {
                "learning_rate": 0.01,
                "linear_hidden_units": [20, 20],
                "final_layer_activation": "None",
                "batch_norm": False,
                "buffer_size": 100000,
                "tau": 0.005,
                "gradient_clipping_norm": 5
            },

            "batch_size": 256,
            "discount_rate": 0.9,
            "mu": 0.0,  # for O-H noise
            "theta": 0.15,  # for O-H noise
            "sigma": 0.25,  # for O-H noise
            "action_noise_std": 0.2,  # for TD3
            "action_noise_clipping_range": 0.5,  # for TD3
            "update_every_n_steps": 20,
            "learning_updates_per_learning_session": 10,

            } ,



        "HIGHER_LEVEL": {

                "Actor": {
                "learning_rate": 0.001,
                "linear_hidden_units": [20, 20],
                "final_layer_activation": "TANH",
                "batch_norm": False,
                "tau": 0.005,
                "gradient_clipping_norm": 5
            },

            "Critic": {
                "learning_rate": 0.01,
                "linear_hidden_units": [20, 20],
                "final_layer_activation": "None",
                "batch_norm": False,
                "buffer_size": 100000,
                "tau": 0.005,
                "gradient_clipping_norm": 5
            },

            "batch_size": 256,
            "discount_rate": 0.9,
            "mu": 0.0,  # for O-H noise
            "theta": 0.15,  # for O-H noise
            "sigma": 0.25,  # for O-H noise
            "action_noise_std": 0.2,  # for TD3
            "action_noise_clipping_range": 0.5,  # for TD3
            "update_every_n_steps": 20,
            "learning_updates_per_learning_session": 10,

            } ,


        }


hiro_agent = HIRO(config)
ll_env = hiro_agent.lower_level_agent.environment
h_env = hiro_agent.higher_level_agent.environment

def test_environment_resets():
    """Tests created environments reset properly"""
    lower_level_state = ll_env.reset()
    assert lower_level_state.shape[0] == 6
    assert ll_env.max_sub_policy_timesteps == 3
    assert ll_env.lower_level_timesteps == 0

    hiro_agent.higher_level_state = np.array([0., 1.0, 2.0])
    hiro_agent.goal = np.array([1.0, 4.0, -22.])
    assert all(ll_env.reset() == np.array([0.0, 1.0, 2.0, 1.0, 4.0, -22.0]))

    high_level_state = h_env.reset()
    assert high_level_state.shape[0] == 3

def test_higher_level_step():
    """Tests environment for higher level steps correctly"""
    h_env.reset()
    assert hiro_agent.higher_level_next_state is None
    next_state, reward, done, _ = h_env.step(np.array([-1.0, 2.0, 3.0]))

    assert all(hiro_agent.goal == np.array([-1.0, 2.0, 3.0]))
    assert all(hiro_agent.higher_level_state == next_state)
    assert all(hiro_agent.higher_level_next_state == next_state)
    assert hiro_agent.higher_level_reward == reward
    assert hiro_agent.higher_level_done == done

    assert next_state.shape[0] == 3
    assert isinstance(reward, float)
    assert not done

    for _ in range(200):
        next_state, reward, done, _ = h_env.step(np.array([-1.0, 2.0, 3.0]))
        assert all(hiro_agent.higher_level_next_state == next_state)
        assert all(hiro_agent.higher_level_next_state == next_state)
        assert hiro_agent.higher_level_reward == reward
        assert hiro_agent.higher_level_done == done

def test_changing_max_lower_timesteps():
    """Tests that changing the max lower level timesteps works"""
    config2 = copy.deepcopy(config)
    config2.hyperparameters["LOWER_LEVEL"]["max_lower_level_timesteps"] = 1
    hiro_agent2 = HIRO(config2)
    h_env2 = hiro_agent2.higher_level_agent.environment
    h_env2.reset()
    next_state, reward, done, _ = h_env2.step(np.array([-1.0, 2.0, 3.0]))

    assert not done
    assert hiro_agent2.lower_level_done
    assert reward == hiro_agent2.higher_level_reward


def test_sub_policy_env_reset():
    """Tests reset method in the sub policy environment we create"""
    external_state = env.reset()
    assert external_state.shape[0] == 6
    assert env.internal_state.shape[0] == 3
    assert all(env.goal == agent.give_next_goal_for_sub_policy())
    assert not env.episode_over
    assert env.timesteps == 0
    assert env.max_sub_policy_timesteps == 5

def test_sub_policy_env_turn_internal_state_to_external_state():
    """Tests turn_internal_state_to_external_state method in the sub policy environment we create"""
    env.goal = np.array([1., 2., 3.])
    result = env.turn_internal_state_to_external_state(np.array([9., 9., 9.]))
    assert all(result == np.array([9., 9., 9., 1., 2., 3.]))
    external_state = env.reset()
    env.goal = np.array([-1., 0., 0.])
    result = env.turn_internal_state_to_external_state(external_state[0:3])
    assert all(result[:3] == external_state[:3]) and all(result[3:] == np.array([-1., 0., 0.]))

def test_sub_policy_env_step():
    """Tests step method in the sub policy environment we create"""
    assert len(agent.extrinsic_rewards) == 0
    external_state = env.reset()
    next_external_state, intrinsic_reward, episode_over, _ = env.step(np.array([0.0]))
    assert next_external_state.shape[0] == 6
    assert all(next_external_state[3:] == external_state[3:])
    assert len(agent.extrinsic_rewards) == 1
    assert env.timesteps == 1
    env.step(np.array([0.0]))
    assert env.timesteps == 2

    external_state = env.reset()
    assert env.timesteps == 0
    env.goal = np.array([0.0, 5.0, -22.0])
    next_external_state, intrinsic_reward, episode_over, _ = env.step(np.array([0.0]))
    assert intrinsic_reward == env.calculate_intrinsic_reward(external_state[0:3], next_external_state[0:3], env.goal)
    assert all(env.internal_next_state == next_external_state[0:3])
    assert env.timesteps == 1
    assert not env.episode_over

    external_state = env.reset()
    for _ in range(4):
        next_external_state, intrinsic_reward, sub_policy_episode_over, _ = env.step(np.array([0.0]))
        assert not env.episode_over
        assert not sub_policy_episode_over

    next_external_state, intrinsic_reward, sub_policy_episode_over, _ = env.step(np.array([0.0]))
    assert sub_policy_episode_over
    assert not env.episode_over

    for _ in range(1000):
        env.step(np.array([0.0]))

    assert env.episode_over
    assert env.sub_policy_episode_over

def test_sub_policy_env_calculate_intrinsic_reward():
    """Tests calculate_intrinsic_reward in the sub policy environment we create"""
    assert env.calculate_intrinsic_reward(5, 9, 4) == 0
    assert env.calculate_intrinsic_reward(5, 9, 1.5) == -(2.5**2)**0.5
    state = np.array([0.5, -3.5])
    next_state = np.array([3.5, 23.5])
    goal = np.array([-23.5, 15.5])
    combined = state + goal - next_state
    result = -np.linalg.norm(combined)
    assert np.allclose(env.calculate_intrinsic_reward(state, next_state, goal), result)



