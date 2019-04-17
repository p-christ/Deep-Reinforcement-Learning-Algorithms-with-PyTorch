from Base_Agent import Base_Agent
from Agents.DQN_Agents.DQN import DQN


class h_DQN(DQN):
    """Implements hierarchical RL agent h-DQN from paper Kulkarni et al. (2016) https://arxiv.org/abs/1604.06057?context=stat"""

    def __init__(self, config):

        DQN.__init__(config)

        self.sg_q_values = self.create_NN(input_dim=self.state_size, output_dim=self.state_size)
        self.sga_q_values = self.create_NN(input_dim=self.state_size * 2, output_dim=self.action_size)

    def step(self):

        # meta controller picks goal
        # controller acts until end of episode or goal achieved

        need_new_subgoal = True

        while not self.done:

            if need_new_subgoal:
                self.subgoal = self.meta_controller_picks_goal()
                self.state =  # add goal to state...
                need_new_subgoal = False

            self.pick_and_conduct_action()
            self.update_next_state_reward_done_and_score()
            if self.time_for_q_network_to_learn():
                self.q_network_learn()
            self.save_experience()
            self.state = self.next_state #this is to set the state for the next iteration

            if self.next_state == self.goal:
                need_new_subgoal = True

            self.global_step_number += 1
        self.episode_number += 1


        # need to save the meta experiences too

    def update_next_state_reward_done_and_score(self):
        """Gets the next state, reward and done information from the environment"""
        self.next_state = self.environment.get_next_state() + append GOAL ...
        self.reward = self.environment.get_reward()
        self.done = self.environment.get_done()
        self.total_episode_score_so_far += self.environment.get_reward()



        pass