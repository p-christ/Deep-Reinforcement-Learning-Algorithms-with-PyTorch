import copy
import numpy as np
from agents.Base_Agent import Base_Agent
from agents.DQN_agents.DDQN import DDQN

class h_DQN(Base_Agent):
    """Implements hierarchical RL agent h-DQN from paper Kulkarni et al. (2016) https://arxiv.org/abs/1604.06057?context=stat
    Note also that this algorithm only works when we have discrete states and discrete actions currently because otherwise
    it is not clear what it means to achieve a subgoal state designated by the meta-controller"""
    agent_name = "h-DQN"

    def __init__(self, config):
        Base_Agent.__init__(self, config)
        self.controller_config = copy.deepcopy(config)
        self.controller_config.hyperparameters = self.controller_config.hyperparameters["CONTROLLER"]
        self.controller = DDQN(self.controller_config)
        self.controller.q_network_local = self.create_NN(input_dim=self.state_size*2, output_dim=self.action_size,
                                                         key_to_use="CONTROLLER")
        self.meta_controller_config = copy.deepcopy(config)
        self.meta_controller_config.hyperparameters = self.meta_controller_config.hyperparameters["META_CONTROLLER"]
        self.meta_controller = DDQN(self.meta_controller_config)
        self.meta_controller.q_network_local = self.create_NN(input_dim=self.state_size, output_dim=config.environment.observation_space.n,
                                                              key_to_use="META_CONTROLLER")
        self.rolling_intrinsic_rewards = []
        self.goals_seen = []
        self.controller_learnt_enough = False
        self.controller_actions = []

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        self.state = self.environment.reset()
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False
        self.cumulative_meta_controller_reward = 0
        self.episode_over = False
        self.subgoal_achieved = False
        self.total_episode_score_so_far = 0
        self.meta_controller_steps = 0
        self.update_learning_rate(self.controller_config.hyperparameters["learning_rate"], self.controller.q_network_optimizer)
        self.update_learning_rate(self.meta_controller_config.hyperparameters["learning_rate"], self.meta_controller.q_network_optimizer)

    def step(self):

        self.episode_steps = 0

        while not self.episode_over:
            episode_intrinsic_rewards = []
            self.meta_controller_state = self.environment.state
            self.subgoal = self.meta_controller.pick_action(state=self.meta_controller_state)
            self.goals_seen.append(self.subgoal)
            self.subgoal_achieved = False
            self.state = np.concatenate((self.environment.state, np.array([self.subgoal])))
            self.cumulative_meta_controller_reward = 0

            while not (self.episode_over or self.subgoal_achieved):
                self.pick_and_conduct_controller_action()
                self.update_data()
                if self.time_to_learn(self.controller.memory, self.global_step_number, "CONTROLLER"): #means it is time to train controller
                    for _ in range(self.hyperparameters["CONTROLLER"]["learning_iterations"]):
                        self.controller.learn()
                self.save_experience(memory=self.controller.memory, experience=(self.state, self.action, self.reward, self.next_state, self.done))
                self.state = self.next_state #this is to set the state for the next iteration
                self.global_step_number += 1
                episode_intrinsic_rewards.append(self.reward)

            if self.time_to_learn(self.meta_controller.memory, self.meta_controller_steps, "META_CONTROLLER"):
                for _ in range(self.hyperparameters["META_CONTROLLER"]["learning_iterations"]):
                    self.meta_controller.learn()

            self.save_experience(memory=self.meta_controller.memory,
                                 experience=(self.meta_controller_state, self.subgoal, self.cumulative_meta_controller_reward,
                                             self.meta_controller_next_state, self.episode_over))
            self.meta_controller_steps += 1
            self.episode_steps += 1

        self.rolling_intrinsic_rewards.append(np.sum(episode_intrinsic_rewards))
        if self.episode_number % 100 == 0:
            print(" ")
            print("Most common goal -- {} -- ".format( max(set(self.goals_seen[-100:]), key=self.goals_seen[-100:].count)  ))
            print("Intrinsic Rewards -- {} -- ".format(np.mean(self.rolling_intrinsic_rewards[-100:])))
            print("Average controller action -- {} ".format(np.mean(self.controller_actions[-100:])))
            print("Latest subgoal -- {}".format(self.goals_seen[-1]))
        self.episode_number += 1
        self.controller.episode_number += 1
        self.meta_controller.episode_number += 1

    def pick_and_conduct_controller_action(self):
        """Picks and conducts an action for controller"""
        self.action =  self.controller.pick_action(state=self.state)
        self.controller_actions.append(self.action)
        self.conduct_action()

    def update_data(self):
        """Updates stored data for controller and meta-controller. It must occur in the order shown"""
        self.episode_over = self.environment.get_done()
        self.update_controller_data()
        self.update_meta_controller_data()

    def update_controller_data(self):
        """Gets the next state, reward and done information from the environment"""
        environment_next_state = self.environment.get_next_state()
        assert environment_next_state.shape[0] == 1
        self.next_state = np.concatenate((environment_next_state, np.array([self.subgoal])))
        self.subgoal_achieved = environment_next_state[0] == self.subgoal
        self.reward = 1.0 * self.subgoal_achieved
        self.done = self.subgoal_achieved or self.episode_over

    def update_meta_controller_data(self):
        """Updates data relating to meta controller"""
        self.cumulative_meta_controller_reward += self.environment.get_reward()
        self.total_episode_score_so_far += self.environment.get_reward()
        if self.done:
            self.meta_controller_next_state = self.environment.get_next_state()

    def time_to_learn(self, memory, steps_taken, controller_name):
        """Boolean indicating whether it is time for meta-controller or controller to learn"""
        enough_experiences = len(memory) > self.hyperparameters[controller_name]["batch_size"]
        enough_steps_taken = steps_taken % self.hyperparameters[controller_name]["update_every_n_steps"] == 0
        return enough_experiences and enough_steps_taken
