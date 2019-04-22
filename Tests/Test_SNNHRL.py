"""Tests for the agent SNNHRL in the Agents folder"""
from Agents.Hierarchical_Agents.SNN_HRL import SNN_HRL
from Utilities.Data_Structures.Config import Config
from Environments.Long_Corridor_Environment import Long_Corridor_Environment
import numpy as np

config = Config()
config.seed = 1
config.num_episodes_to_run = 10

config.hyperparameters = {
        "SKILL_AGENT": {
            "num_skills": 11,
            "regularisation_weight": 0.5,
            "visitations_decay": 1.0,
            "episodes_for_pretraining": 100
    },
    "MANAGER": {
        "timesteps_before_changing_skill": 5
    }
}

def test_state_visitations():
    """Tests the state visitations data structure works properly"""
    config.environment = Long_Corridor_Environment()
    config.env_parameters = {"stochasticity_of_action_right": 0.0}
    agent = SNN_HRL(config)
    skill_agent = agent.create_skill_training_agent()
    # agent.skills_env =
    state_visitations = skill_agent.environment.state_visitations
    assert len(state_visitations) == 10
    assert len(state_visitations[0]) == 6

def test_reset():
    """Tests the reset function works properly in subclass"""
    config.environment = Long_Corridor_Environment()
    config.env_parameters = {"stochasticity_of_action_right": 0.0}
    agent = SNN_HRL(config)

    state = agent.skills_env.reset()
    assert all(state == np.array([1.0, 10]))

def test_step():
    """Tests the step function works properly in subclass"""
    config.environment = Long_Corridor_Environment()
    config.env_parameters = {"stochasticity_of_action_right": 0.0}
    agent = SNN_HRL(config)

    agent.skills_env.reset()

    next_state, reward, done, _ = agent.skills_env.step(1)
    assert all(next_state == np.array([2, 10]))
    assert reward == 0
    assert not done

    next_state, reward, done, _ = agent.skills_env.step(1)
    assert all(next_state == np.array([3, 10]))
    assert reward == 0
    assert not done

    agent.skill = 9

    assert all(next_state == np.array([3, 10]))

    next_state, reward, done, _ = agent.skills_env.step(0)
    assert all(next_state == np.array([2, 9]))
    assert np.allclose(reward,  -0.34657359028)
    assert not done







def test_skills_environment_state_size_change():
    """Tests whether the change to the state_size for the skill environment occurs correctly"""
    config.environment = Long_Corridor_Environment()
    config.env_parameters = {"stochasticity_of_action_right": 0.0}
    agent = SNN_HRL(config)
    assert agent.skill_agent_config.environment.state == 2


    # ADD FOUR ROOMS ENVIRONMENT TO TESTS....

def test_skills_environment_get_state_change():
    """Tests whether the change to the get_state for the skill environment occurs correctly"""
    config.environment = Long_Corridor_Environment(stochasticity_of_action_right=0.0)
    config.env_parameters = {"stochasticity_of_action_right": 0.0}

    agent = SNN_HRL(config)
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

    agent = SNN_HRL(config)
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
