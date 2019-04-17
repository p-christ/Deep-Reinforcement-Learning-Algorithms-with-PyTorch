import torch.optim as optim
import numpy as np
from Agents.DQN_Agents.DQN import DQN
from Replay_Buffer import Replay_Buffer


class h_DQN(DQN):
    """Implements hierarchical RL agent h-DQN from paper Kulkarni et al. (2016) https://arxiv.org/abs/1604.06057?context=stat"""

    def __init__(self, config):
        DQN.__init__(config)

        self.q_network_local = self.create_NN(input_dim=self.state_size*2, output_dim=self.action_size)

        self.meta_controller_memory = Replay_Buffer(self.hyperparameters["buffer_size"], self.hyperparameters["batch_size"], config.seed)
        self.meta_controller_q_network_local = self.create_NN(input_dim=self.state_size, output_dim=self.config.environment.get_num_possible_states())
        self.meta_controller_q_optimizer = optim.Adam(self.q_network_local.parameters(),
                                              lr=self.hyperparameters["meta_controller_learning_rate"])

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        self.state = self.environment.reset_environment()
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False

        self.intrinsic_reward = None
        self.subgoal_achieved = False
        self.cumulative_meta_controller_reward = 0

        self.total_episode_score_so_far = 0
        self.episode_states = []
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_next_states = []
        self.episode_dones = []


    def step(self):

        while not self.done:

            self.meta_controller_state = self.environment.get_state()
            self.subgoal = self.meta_controller_picks_goal()
            self.state = np.concatenate((self.state, self.subgoal))
            self.cumulative_meta_controller_reward = 0

            while not self.done or not self.subgoal_achieved:
                self.pick_and_conduct_action()
                self.update_controller_data()
                self.update_meta_controller_data()
                self.save_controller_experience()
                self.state = self.next_state #this is to set the state for the next iteration
                self.global_step_number += 1
            self.save_meta_controller_experience()
        self.episode_number += 1


    def update_controller_next_state_reward_done_and_score(self):
        """Gets the next state, reward and done information from the environment"""
        self.next_state = np.concatenate((self.environment.get_next_state(), self.subgoal))
        self.intrinsic_reward = 1 if self.environment.get_next_state() == self.subgoal else 0
        self.subgoal_achieved = True if self.environment.get_next_state() == self.subgoal else 0

    def update_meta_controller_data(self):

        self.cumulative_meta_controller_reward += self.environment.get_reward()
        self.total_episode_score_so_far += self.environment.get_reward()
        self.done = self.environment.get_done()

        if self.subgoal_achieved or self.done:
            self.meta_controller_next_state = self.environment.get_next_state()

    
    def save_experience(self):
        """Saves the recent experience to the memory buffer"""
        self.memory.add_experience(self.state, self.action, self.reward, self.next_state, self.done)





        pass