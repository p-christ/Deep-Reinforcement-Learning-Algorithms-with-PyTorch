from Utilities.Utility_Functions import abstract

@abstract
class HER_Base(object):
    """Contains methods needed to turn an algorithm into a hindsight experience replay (HER) algorithm"""
    def __init__(self):
        raise ValueError("The HER_Extension is not to be instantiated, only to be inherited")

    def save_alternative_experience(self):
        """Saves the experiences as if the final state visited in the episode was the goal state"""
        dimension_of_goal = len(self.environment.desired_goal)
        new_goal = self.environment.get_achieved_goal()
        num_observations = len(self.episode_states)
        reward_for_achieving_goal = self.environment.get_reward_for_achieving_goal()
        step_reward_for_not_achieving_goal = self.environment.get_step_reward_for_not_achieving_goal()

        new_states = [self.swap_state_goal_for_achieved_goal(state, dimension_of_goal, new_goal) for state in self.episode_states]
        new_next_states = [self.swap_state_goal_for_achieved_goal(next_state, dimension_of_goal, new_goal) for next_state in
                      self.episode_next_states]
        new_rewards = [step_reward_for_not_achieving_goal if ix != num_observations - 1 else reward_for_achieving_goal for ix in range(num_observations)]

        self.memory.add_experience(new_states, self.episode_actions, new_rewards, new_next_states, self.episode_dones)

    def swap_state_goal_for_achieved_goal(self, state, dimension_of_goal, new_goal):
        """Swaps in the achieved goal for the desired goal in the state representation"""
        state[-dimension_of_goal:] = new_goal
        return state


