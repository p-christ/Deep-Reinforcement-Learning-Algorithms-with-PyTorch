"""Tests for the hierarchical RL agent HIRO"""
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
config.num_episodes_to_run = 2000
config.file_to_save_data_results = None
config.file_to_save_results_graph = None
config.visualise_individual_results = False
config.visualise_overall_agent_results = False
config.randomise_random_seed = False
config.runs_per_agent = 1
config.use_GPU = False
config.hyperparameters = {

    "HIRO": {
        "max_sub_policy_timesteps": 5
    }
}

config.hyperparameters = config.hyperparameters["HIRO"]
agent = HIRO(config)
env = agent.env_for_sub_policy

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



