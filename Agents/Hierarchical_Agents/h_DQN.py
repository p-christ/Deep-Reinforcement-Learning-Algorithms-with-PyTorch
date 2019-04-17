import torch
import torch.optim as optim
import numpy as np
from Agents.DQN_Agents.DQN import DQN
from Base_Agent import Base_Agent
from Replay_Buffer import Replay_Buffer

class h_DQN(DQN):
    """Implements hierarchical RL agent h-DQN from paper Kulkarni et al. (2016) https://arxiv.org/abs/1604.06057?context=stat
    Note that we inherit from DQN which predominantly takes care of handling the controller. For the meta-controller we write extra code.
    Note also that this algorithm only works when we have discrete states and discrete actions currently because otherwise
    it is not clear what it means to achieve a subgoal state designated by the meta-controller"""

    agent_name = "h_DQN"

    def __init__(self, config):

        Base_Agent.__init__(self, config)

        print(self.hyperparameters)

        self.controller_memory = Replay_Buffer(self.hyperparameters["CONTROLLER"]["buffer_size"],
                                               self.hyperparameters["CONTROLLER"]["batch_size"], config.seed)
        self.controller_q_network_local = self.create_NN(input_dim=self.state_size*2, output_dim=self.action_size, key_to_use="CONTROLLER")
        self.controller_q_network_optimizer = optim.Adam(self.controller_q_network_local.parameters(),
                                              lr=self.hyperparameters["CONTROLLER"]["learning_rate"])

        self.meta_controller_memory = Replay_Buffer(self.hyperparameters["META_CONTROLLER"]["buffer_size"],
                                                    self.hyperparameters["META_CONTROLLER"]["batch_size"], config.seed)
        self.meta_controller_q_network_local = self.create_NN(input_dim=self.state_size, output_dim=self.config.environment.get_num_possible_states(),
                                                              key_to_use="META_CONTROLLER")
        self.meta_controller_q_optimizer = optim.Adam(self.meta_controller_q_network_local.parameters(),
                                              lr=self.hyperparameters["META_CONTROLLER"]["learning_rate"])

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        self.state = self.environment.reset_environment()
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False

        self.cumulative_meta_controller_reward = 0

        self.subgoal_achieved = False
        self.cumulative_meta_controller_reward = 0
        self.episode_over = False

        self.total_episode_score_so_far = 0
        self.episode_states = []
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_next_states = []
        self.episode_dones = []

        self.meta_controller_steps = 0


    def step(self):

        while not self.episode_over:

            self.meta_controller_state = self.environment.get_state()
            self.subgoal = self.meta_controller_picks_goal()
            self.subgoal_achieved = False
            print("Meta controller goal ", self.subgoal)

            self.state = np.concatenate((self.environment.get_state(), np.array([self.subgoal])))
            print("Self state ", self.state)
            self.cumulative_meta_controller_reward = 0

            while not self.episode_over and not self.subgoal_achieved:
                self.pick_and_conduct_controller_action()
                self.update_data()

                if self.time_for_controller_to_learn(): #means it is time to train controller
                    self.q_network_learn(q_network=self.controller_q_network_local, optimizer=self.controller_q_network_optimizer,
                                         replay_buffer=self.controller_memory, start_learning_rate=self.hyperparameters["CONTROLLER"]["learning_rate"])
                self.save_experience(memory=self.controller_memory)
                print("NEXT STATE ", self.environment.get_next_state())
                self.state = self.next_state #this is to set the state for the next iteration
                self.global_step_number += 1

            if self.time_for_meta_controller_to_learn():
                self.q_network_learn(q_network=self.meta_controller_q_network_local,
                                     optimizer=self.meta_controller_q_network_optimizer,
                                     replay_buffer=self.meta_controller_memory,
                                     start_learning_rate=self.hyperparameters["META_CONTROLLER"]["learning_rate"])
            print("Final state ", self.environment.get_next_state())
            print("Achieved goal or not ", self.reward)

            self.save_meta_controller_experience()
            self.meta_controller_steps += 1
        self.episode_number += 1

    def pick_and_conduct_controller_action(self):
        """Picks and conducts an action for controller"""
        self.action = self.pick_action(q_network=self.controller_q_network_local,
                                       epsilon_decay_denominator=self.hyperparameters["CONTROLLER"]["epsilon_decay_denominator"])
        self.conduct_action()

    def meta_controller_picks_goal(self):
        """Meta controller uses epsilon greedy policy on its q_function to pick amongst subgoals for the controller
        to try and achieve"""
        state = torch.from_numpy(self.meta_controller_state).float().unsqueeze(0).to(self.device)
        self.meta_controller_q_network_local.eval() #puts network in evaluation mode
        with torch.no_grad():
            action_values = self.meta_controller_q_network_local(state)
        self.meta_controller_q_network_local.train() #puts network back in training mode
        subgoal = self.make_epsilon_greedy_choice(action_values, self.hyperparameters["META_CONTROLLER"]["epsilon_decay_denominator"])
        return subgoal

    def time_for_meta_controller_to_learn(self):
        """Boolean indicating whether it is time for meta-controller to learn"""
        enough_experiences = len(self.meta_controller_memory) > self.hyperparameters["META_CONTROLLER"]["batch_size"]
        enough_steps_taken = self.meta_controller_steps % self.hyperparameters["META_CONTROLLER"]["update_every_n_steps"] == 0
        return enough_experiences and enough_steps_taken

    def time_for_controller_to_learn(self):
        """Boolean indicating whether it is time for controller to learn"""
        enough_experiences = len(self.meta_controller_memory) > self.hyperparameters["CONTROLLER"]["batch_size"]
        enough_steps_taken = self.meta_controller_steps % self.hyperparameters["CONTROLLER"]["update_every_n_steps"] == 0
        return enough_experiences and enough_steps_taken


    def update_data(self):
        """Updates stored data for controller and meta-controller. It must occur in the order shown"""
        self.episode_over = self.environment.get_done()
        print("EPISODE OVER ", self.episode_over)
        self.update_controller_data()
        self.update_meta_controller_data()

    def update_controller_data(self):
        """Gets the next state, reward and done information from the environment"""
        self.next_state = np.concatenate((self.environment.get_next_state(), np.array([self.subgoal])))
        self.reward = 1 if self.environment.get_next_state() == self.subgoal else 0

        self.subgoal_achieved = self.environment.get_next_state() == self.subgoal
        self.done = True if self.environment.get_next_state() == self.subgoal or self.episode_over else False

    def update_meta_controller_data(self):

        self.cumulative_meta_controller_reward += self.environment.get_reward()
        self.total_episode_score_so_far += self.environment.get_reward()
        if self.done:
            self.meta_controller_next_state = self.environment.get_next_state()

    def save_meta_controller_experience(self):
        """Saves the recent experience to memory buffer for meta_controller"""
        self.meta_controller_memory.add_experience(self.meta_controller_state, self.subgoal, self.cumulative_meta_controller_reward,
                                                   self.meta_controller_next_state, self.episode_over)
