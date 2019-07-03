import random
from collections import Counter

import pytest

from agents.DQN_agents.DQN_HER import DQN_HER
from agents.DQN_agents.DDQN import DDQN
from agents.DQN_agents.DDQN_With_Prioritised_Experience_Replay import DDQN_With_Prioritised_Experience_Replay
from agents.DQN_agents.DQN_With_Fixed_Q_Targets import DQN_With_Fixed_Q_Targets
from environments.Bit_Flipping_Environment import Bit_Flipping_Environment
from agents.policy_gradient_agents.PPO import PPO
from agents.Trainer import Trainer
from utilities.data_structures.Config import Config
from agents.DQN_agents.DQN import DQN
import numpy as np
import torch

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

config = Config()
config.seed = 1
config.environment = Bit_Flipping_Environment(4)
config.num_episodes_to_run = 1
config.file_to_save_data_results = None
config.file_to_save_results_graph = None
config.visualise_individual_results = False
config.visualise_overall_agent_results = False
config.randomise_random_seed = False
config.runs_per_agent = 1
config.use_GPU = False
config.hyperparameters = {

    "DQN_Agents": {

        "learning_rate": 0.005,
        "batch_size": 3,
        "buffer_size": 40000,
        "epsilon": 0.1,
        "epsilon_decay_rate_denominator": 200,
        "discount_rate": 0.99,
        "tau": 0.1,
        "alpha_prioritised_replay": 0.6,
        "beta_prioritised_replay": 0.4,
        "incremental_td_error": 1e-8,
        "update_every_n_steps": 3,
        "linear_hidden_units": [20, 20, 20],
        "final_layer_activation": "None",
        "batch_norm": False,
        "gradient_clipping_norm": 5,
        "HER_sample_proportion": 0.8,
        "clip_rewards": False
}
}


trainer = Trainer(config, [DQN_HER])
config.hyperparameters = config.hyperparameters["DQN_Agents"]
agent = DQN_HER(config)
agent.reset_game()

def test_initiation():
    """Tests whether DQN_HER initiates correctly"""
    config.hyperparameters["batch_size"] = 64
    agent = DQN_HER(config)
    agent.reset_game()


    assert agent.ordinary_buffer_batch_size == int(0.2 * 64)
    assert agent.HER_buffer_batch_size == 64 - int(0.2 * 64)

    assert agent.q_network_local.input_dim == 8
    assert agent.q_network_local.output_layers[0].out_features == 4

    assert isinstance(agent.state_dict, dict)

    assert agent.observation.shape[0] == 4
    assert agent.desired_goal.shape[0] == 4
    assert agent.achieved_goal.shape[0] == 4

    assert agent.state.shape[0] == 8
    assert not agent.done
    assert agent.next_state is None
    assert agent.reward is None

    config.hyperparameters["batch_size"] = 3

def test_action():
    """Tests whether DQN_HER picks and conducts actions correctly"""
    num_tries = 1000
    actions = []
    for _ in range(num_tries):
        action = agent.pick_action()
        actions.append(action)

    actions_count = Counter(actions)
    assert actions_count[0] > num_tries*0.1
    assert actions_count[1] > num_tries*0.1
    assert actions_count[2] > num_tries*0.1
    assert actions_count[3] > num_tries*0.1
    assert actions_count[0] + actions_count[1] + actions_count[2] + actions_count[3] == num_tries

    assert agent.next_state is None

def test_tracks_changes_from_one_action():
    """Tests that it tracks the changes as a result of actions correctly"""

    previous_obs = agent.observation
    previous_desired_goal = agent.desired_goal
    previous_achieved_goal = agent.achieved_goal

    agent.action = 0
    agent.conduct_action_in_changeable_goal_envs(agent.action)

    assert agent.next_state.shape[0] == 8
    assert isinstance(agent.next_state_dict, dict)
    assert not all (agent.observation == previous_obs)
    assert not all(agent.achieved_goal == previous_achieved_goal)
    assert all (agent.desired_goal == previous_desired_goal)

    agent.track_changeable_goal_episodes_data()

    with pytest.raises(Exception):
        agent.HER_memory.sample(1)

    agent.save_alternative_experience()

    sample = agent.HER_memory.sample(1)

    assert sample[1].item() == agent.action
    assert sample[2].item() == 4

def test_tracks_changes_from_multiple_actions():
    """Tests that it tracks the changes as a result of actions correctly"""

    agent = DQN_HER(config)
    agent.reset_game()

    for ix in range(4):
        previous_obs = agent.observation
        previous_desired_goal = agent.desired_goal
        previous_achieved_goal = agent.achieved_goal

        agent.action = ix
        agent.conduct_action_in_changeable_goal_envs(agent.action)

        assert agent.next_state.shape[0] == 8
        assert isinstance(agent.next_state_dict, dict)
        assert not all(agent.observation == previous_obs)
        assert not all(agent.achieved_goal == previous_achieved_goal)
        assert all(agent.desired_goal == previous_desired_goal)

        agent.track_changeable_goal_episodes_data()
        agent.save_experience()
        if agent.done: agent.save_alternative_experience()

        agent.state_dict = agent.next_state_dict  # this is to set the state for the next iteration
        agent.state = agent.next_state

    states, actions, rewards, next_states, dones = agent.HER_memory.sample(4)

    assert all(states[1] == torch.Tensor([1.0, 1., 1., 1., 0., 0., 0. , 0.]))
    assert all(actions == torch.Tensor([[1.], [0.], [3.], [2.]]))
    assert all(rewards == torch.Tensor([[-1.], [-1.], [4.], [-1.]]))
    assert all(dones == torch.Tensor([[0.], [0.], [1.], [0.]]))







