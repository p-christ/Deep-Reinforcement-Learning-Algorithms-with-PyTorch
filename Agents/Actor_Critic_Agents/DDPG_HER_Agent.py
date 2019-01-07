from DDPG_Agent import DDPG_Agent


# WIP not finished

#TODO override the save_experience and step method so that it only saves at end of episode and saves two types of experience,
#one with the normal goal and one as if the goal had been the state we achieved in the final timestep


class DDPG_HER_Agent(DDPG_Agent):
    agent_name = "DDPG_HER"

    def __init__(self, config):
        DDPG_Agent.__init__(self, config)

    def step(self):
        """Runs a step within a game including a learning step if required"""
        self.pick_and_conduct_action()
        self.update_next_state_reward_done_and_score()

        if self.time_for_critic_and_actor_to_learn():
            for _ in range(self.ddpg_hyperparameters["learning_updates_per_learning_session"]):
                states, actions, rewards, next_states, dones = self.sample_experiences()  # Sample experiences
                self.critic_learn(experiences_given=True, experiences=(states, actions, rewards, next_states, dones))
                self.actor_learn(states)
        self.track_episodes_data()
        self.save_experience()


        if self.done:
            self.save_alternative_experience()

        self.state = self.next_state #this is to set the state for the next iteration

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

        for ix in range(num_observations):
            new_state = self.episode_states[ix]
            new_state[-dimension_of_goal:] = new_goal

            new_next_state = self.episode_next_states[ix]
            new_next_state[-dimension_of_goal:] = new_goal

            if ix == num_observations - 1:
                new_reward = self.environment.get_reward_for_achieving_goal()
            else:
                new_reward = self.get_step_reward_for_not_achieving_goal()

            self.memory.add_experience(new_state, self.episode_actions[ix], new_reward, new_next_state, self.episode_dones[ix])

