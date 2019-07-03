"""Tests for the hierarchical RL agent HIRO"""
import copy

import gym
import random
import numpy as np
import torch

from agents.hierarchical_agents.HIRO import HIRO
from utilities.data_structures.Config import Config

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
                "gradient_clipping_norm": 5,

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
            "number_goal_candidates": 8,
            "clip_rewards": False


            } ,



        "HIGHER_LEVEL": {

                "Actor": {
                "learning_rate": 0.001,
                "linear_hidden_units": [20, 20],
                "final_layer_activation": "TANH",
                "batch_norm": False,
                "tau": 0.005,
                "gradient_clipping_norm": 5,
                "number_goal_candidates": 8
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
            "number_goal_candidates": 8,
            "clip_rewards": False

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


def test_goal_transition():
    """Tests environment does goal transitions properly"""
    hiro_agent.higher_level_state = 2
    hiro_agent.goal = 9
    next_state = 3
    assert HIRO.goal_transition(hiro_agent.higher_level_state, hiro_agent.goal, next_state) == 8

    hiro_agent.higher_level_state = 2
    hiro_agent.goal = 9
    next_state = 3
    ll_env.update_goal(next_state)
    assert hiro_agent.goal == 8

    h_env.reset()
    hiro_agent.goal = np.array([2.0, 4.0, -3.0])
    hiro_agent.higher_level_reward = 0
    ll_env.reset()
    state = hiro_agent.higher_level_state
    next_state, reward, done, _ = ll_env.step(np.array([random.random()]))
    assert all(hiro_agent.goal == state + np.array([2.0, 4.0, -3.0]) - next_state[0:3])



def test_higher_level_step():
    """Tests environment for higher level steps correctly"""
    hiro_agent = HIRO(config)
    ll_env = hiro_agent.lower_level_agent.environment
    h_env = hiro_agent.higher_level_agent.environment
    h_env.reset()
    # HIRO.goal_transition = lambda x, y, z: y
    state_before = hiro_agent.higher_level_state
    assert hiro_agent.higher_level_next_state is None
    next_state, reward, done, _ = h_env.step(np.array([-1.0, 2.0, 3.0]))

    assert np.allclose(hiro_agent.goal, HIRO.goal_transition(state_before,  np.array([-1.0, 2.0, 3.0]), next_state))

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

def test_lower_level_step():
    """Tests the step level for the lower level environment"""
    ll_env.reset()
    h_env.reset()
    state = ll_env.reset()
    assert state.shape[0] == 6

    hl_next_state, reward, done, _ = h_env.step(np.array([-1.0, 2.0, 3.0]))

    assert all(hl_next_state == hiro_agent.lower_level_next_state[:3])
    assert all(hl_next_state == hiro_agent.lower_level_state[:3])
    assert hiro_agent.lower_level_done
    assert ll_env.lower_level_timesteps == 3

    previous_goal = hiro_agent.goal

    next_state, reward, done, _ = ll_env.step(np.array([-1.0]))

    assert next_state.shape[0] == 6
    assert all(next_state[3:] == hiro_agent.goal)
    assert all(next_state[:3] == hiro_agent.higher_level_next_state)
    assert all(next_state[:3] == hiro_agent.higher_level_state)
    assert all(next_state == hiro_agent.lower_level_next_state)
    assert all(next_state == hiro_agent.lower_level_state)
    assert done == hiro_agent.lower_level_done
    assert not hiro_agent.higher_level_done
    assert reward == ll_env.calculate_intrinsic_reward(hl_next_state,  hiro_agent.higher_level_next_state , previous_goal)

    for _ in range(100):
        previous_goal = hiro_agent.goal
        state = hiro_agent.higher_level_next_state
        next_state, reward, done, _ = ll_env.step(np.array([random.random()]))
        assert next_state.shape[0] == 6
        assert all(next_state[3:] == hiro_agent.goal)
        assert all(next_state[:3] == hiro_agent.higher_level_next_state)
        assert all(next_state[:3] == hiro_agent.higher_level_state)
        assert all(next_state == hiro_agent.lower_level_next_state)
        assert all(next_state == hiro_agent.lower_level_state)
        assert done == hiro_agent.lower_level_done
        assert reward == ll_env.calculate_intrinsic_reward(state, hiro_agent.higher_level_next_state,
                                                           previous_goal)

def test_sub_policy_env_turn_internal_state_to_external_state():
    """Tests turn_internal_state_to_external_state method in the sub policy environment we create"""
    goal = np.array([1., 2., 3.])
    result = ll_env.turn_internal_state_to_external_state(np.array([9., 9., 9.]), goal)
    assert all(result == np.array([9., 9., 9., 1., 2., 3.]))
    external_state = ll_env.reset()
    goal = np.array([-1., 0., 0.])
    result = ll_env.turn_internal_state_to_external_state(external_state[0:3], goal)
    assert all(result[:3] == external_state[:3]) and all(result[3:] == np.array([-1., 0., 0.]))
