import torch
import numpy as np
from utilities.data_structures.Replay_Buffer import Replay_Buffer
from utilities.Utility_Functions import abstract

@abstract
class HER_Base(object):
    """Contains methods needed to turn an algorithm into a hindsight experience replay (HER) algorithm"""
    def __init__(self, buffer_size, batch_size, HER_sample_proportion):
        self.HER_memory = Replay_Buffer(buffer_size, batch_size, self.config.seed)
        self.ordinary_buffer_batch_size = int(batch_size * (1.0 - HER_sample_proportion))
        self.HER_buffer_batch_size = batch_size - self.ordinary_buffer_batch_size

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        self.state_dict = self.environment.reset()
        self.observation = self.state_dict["observation"]
        self.desired_goal = self.state_dict["desired_goal"]
        self.achieved_goal = self.state_dict["achieved_goal"]

        self.state = self.create_state_from_observation_and_desired_goal(self.observation, self.desired_goal)
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False

        self.episode_states = []
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_next_states = []
        self.episode_dones = []

        self.episode_desired_goals = []
        self.episode_achieved_goals = []
        self.episode_observations = []

        self.episode_next_desired_goals = []
        self.episode_next_achieved_goals = []
        self.episode_next_observations = []

        self.total_episode_score_so_far = 0

    def track_changeable_goal_episodes_data(self):
        """Saves the data from the recent episodes in a way compatible with changeable goal environments"""
        self.episode_rewards.append(self.reward)
        self.episode_actions.append(self.action)
        self.episode_dones.append(self.done)

        self.episode_states.append(self.state)
        self.episode_next_states.append(self.next_state)

        self.episode_desired_goals.append(self.state_dict["desired_goal"])
        self.episode_achieved_goals.append(self.state_dict["achieved_goal"])
        self.episode_observations.append(self.state_dict["observation"])

        self.episode_next_desired_goals.append(self.next_state_dict["desired_goal"])
        self.episode_next_achieved_goals.append(self.next_state_dict["achieved_goal"])
        self.episode_next_observations.append(self.next_state_dict["observation"])

    def conduct_action_in_changeable_goal_envs(self, action):
        """Adapts conduct_action from base agent so that can handle changeable goal environments"""
        self.next_state_dict, self.reward, self.done, _ = self.environment.step(action)
        self.total_episode_score_so_far += self.reward
        if self.hyperparameters["clip_rewards"]:
            self.reward = max(min(self.reward, 1.0), -1.0)
        self.observation = self.next_state_dict["observation"]
        self.desired_goal = self.next_state_dict["desired_goal"]
        self.achieved_goal = self.next_state_dict["achieved_goal"]
        self.next_state =  self.create_state_from_observation_and_desired_goal(self.observation, self.desired_goal)


    def create_state_from_observation_and_desired_goal(self, observation, desired_goal):
        return np.concatenate((observation, desired_goal))

    def save_alternative_experience(self):
        """Saves the experiences as if the final state visited in the episode was the goal state"""
        new_goal = self.achieved_goal
        new_states = [self.create_state_from_observation_and_desired_goal(observation, new_goal) for observation in self.episode_observations]
        new_next_states = [self.create_state_from_observation_and_desired_goal(observation, new_goal) for observation in
                      self.episode_next_observations]
        new_rewards = [self.environment.compute_reward(next_achieved_goal, new_goal, None) for next_achieved_goal in  self.episode_next_achieved_goals]

        if self.hyperparameters["clip_rewards"]:
            new_rewards = [max(min(reward, 1.0), -1.0) for reward in new_rewards]

        self.HER_memory.add_experience(new_states, self.episode_actions, new_rewards, new_next_states, self.episode_dones)

    def sample_from_HER_and_Ordinary_Buffer(self):
        """Samples from the ordinary replay buffer and HER replay buffer according to a proportion specified in config"""
        states, actions, rewards, next_states, dones = self.memory.sample(self.ordinary_buffer_batch_size)
        HER_states, HER_actions, HER_rewards, HER_next_states, HER_dones = self.HER_memory.sample(self.HER_buffer_batch_size)

        states = torch.cat((states, HER_states))
        actions = torch.cat((actions, HER_actions))
        rewards = torch.cat((rewards, HER_rewards))
        next_states = torch.cat((next_states, HER_next_states))
        dones = torch.cat((dones, HER_dones))
        return states, actions, rewards, next_states, dones


