"""Tests for the agent SNNHRL in the Agents folder"""
from Environments.Bit_Flipping_Environment import Bit_Flipping_Environment
from Agents.Hierarchical_Agents.SNNHRL import SNNHRL
from Utilities.Data_Structures.Config import Config
from Environments.Long_Corridor_Environment import Long_Corridor_Environment
import numpy as np

config = Config()
config.seed = 1

config.hyperparameters = {
        "SKILL_AGENT": {
            "num_skills": 10
    }
}


def test_skills_environment_state_size_change():
    """Tests whether the change to the state_size for the skill environment occurs correctly"""
    config.environment = Long_Corridor_Environment()
    config.env_parameters = {"stochasticity_of_action_right": 0.0}
    agent = SNNHRL(config)
    assert agent.skill_agent_config.environment.get_state_size() == 2

def test_skills_environment_get_state_change():
    """Tests whether the change to the get_state for the skill environment occurs correctly"""
    config.environment = Long_Corridor_Environment(stochasticity_of_action_right=0.0)
    config.env_parameters = {"stochasticity_of_action_right": 0.0}

    agent = SNNHRL(config)
    assert np.array_equal(agent.skill_agent_config.environment.get_state(), np.array([1, 10]))

    agent.skill = 12
    assert np.array_equal(agent.skill_agent_config.environment.get_state(), np.array([1, 12]))

    agent.skill_agent_config.environment.conduct_action(1)
    assert np.array_equal(agent.skill_agent_config.environment.get_state(), np.array([2, 12]))

    agent.skill_agent_config.environment.conduct_action(1)
    agent.skill_agent_config.environment.conduct_action(1)
    agent.skill_agent_config.environment.conduct_action(1)
    assert np.array_equal(agent.skill_agent_config.environment.get_state(), np.array([5, 12]))


def test_skills_environment_get_next_state_change():
    """Tests whether the change to the get_next_state for the skill environment occurs correctly"""
    config.environment = Long_Corridor_Environment(stochasticity_of_action_right=0.0)
    config.env_parameters = {"stochasticity_of_action_right": 0.0}

    agent = SNNHRL(config)
    agent.skill_agent_config.environment.conduct_action(1)
    assert np.array_equal(agent.skill_agent_config.environment.get_next_state(), np.array([2, 10]))

    agent.skill = 12
    assert np.array_equal(agent.skill_agent_config.environment.get_next_state(), np.array([2, 12]))

    agent.skill_agent_config.environment.conduct_action(1)
    assert np.array_equal(agent.skill_agent_config.environment.get_next_state(), np.array([3, 12]))

    agent.skill_agent_config.environment.conduct_action(1)
    agent.skill_agent_config.environment.conduct_action(1)
    agent.skill_agent_config.environment.conduct_action(0)
    assert np.array_equal(agent.skill_agent_config.environment.get_next_state(), np.array([4, 12]))


    #
    #

    # config.environment = Bit_Flipping_Environment()
    # agent = SNNHRL(config)
    # assert agent.skill_agent_config.environment.get_state() == 41
    #
