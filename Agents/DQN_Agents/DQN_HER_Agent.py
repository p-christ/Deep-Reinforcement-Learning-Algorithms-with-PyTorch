from DQN_Agent import DQN_Agent


class DQN_HER_Agent(DQN_Agent):

    agent_name = "DQN_HER"

    def __init__(self, config):
        DQN_Agent.__init__(self, config)

    def step(self):
        """Runs a step within a game including a learning step if required"""
        self.pick_and_conduct_action()
        self.update_next_state_reward_done_and_score()
        if self.time_for_critic_to_learn():
            self.critic_learn()
        self.track_episodes_data()
        self.save_experience()


        if self.done:
            self.save_alternative_experience()

        self.state = self.next_state  # this is to set the state for the next iteration

    def track_episodes_data(self):
        self.episode_states.append(self.state)
        self.episode_actions.append(self.action)
        self.episode_next_states.append(self.next_state)
        self.episode_dones.append(self.done)

    def save_alternative_experience(self):
        """Saves the experiences as if the final state visited in the episode was the goal state"""

        dimension_of_goal = len(self.environment.desired_goal)
        new_goal = self.environment.get_achieved_goal()
        num_observations = len(self.episode_states)

        reward_for_achieving_goal = self.environment.get_reward_for_achieving_goal()
        step_reward_for_not_achieving_goal = self.environment.get_step_reward_for_not_achieving_goal()

        for ix in range(num_observations):
            new_state = self.episode_states[ix]
            new_state[-dimension_of_goal:] = new_goal

            new_next_state = self.episode_next_states[ix]
            new_next_state[-dimension_of_goal:] = new_goal

            if ix == num_observations - 1:
                new_reward = reward_for_achieving_goal
            else:
                new_reward = step_reward_for_not_achieving_goal

            self.memory.add_experience(new_state, self.episode_actions[ix], new_reward, new_next_state,
                                       self.episode_dones[ix])

