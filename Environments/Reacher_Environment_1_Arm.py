from Environments.Base_Environment import Base_Environment
from unityagents import UnityEnvironment


class Reacher_Environment_1_Arm(Base_Environment):

    def __init__(self):
        self.game_environment = UnityEnvironment("/Users/petroschristodoulou/Documents/Deep_RL_Implementations/deep-reinforcement-learning/p2_continuous-control/Reacher_Linux_NoVis/Reacher.x86_64")
        self.brain_name = self.game_environment.brain_names[0]
        self.brain = self.game_environment.brains[self.brain_name]
        self.game_environment_info = self.game_environment.reset(train_mode=True)[self.brain_name]

    def conduct_action(self, action):
        self.game_environment_info = self.game_environment.step(action)[self.brain_name]

    def get_action_size(self):
        return self.brain.vector_action_space_size

    def get_state_size(self):
        return len(self.game_environment_info.vector_observations[0])

    def get_state(self):
        return self.game_environment_info.vector_observations[0]

    def get_next_state(self):
        return self.game_environment_info.vector_observations[0]

    def get_reward(self):
        return self.game_environment_info.rewards[0]

    def get_done(self):
        return self.game_environment_info.local_done[0]

    def reset_environment(self):
        self.game_environment_info = self.game_environment.reset(train_mode=True)[self.brain_name]

    def get_max_steps_per_episode(self):
        return 1000

    def get_action_types(self):
        return "CONTINUOUS"

    def get_rolling_period_to_calculate_score_over(self):
        return 100

    def get_score_to_win(self):
        return 30
