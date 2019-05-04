from Base_Agent import Base_Agent
import copy
import time

from DDQN import DDQN


class HRL(DDQN):
    agent_name = "HRL"

    def __init__(self, config):
        super().__init__(config)
        # self.agent_config = copy.deepcopy(config)
        # # self.agent_config.environment = DIAYN_Skill_Wrapper(self.agent_config.environment)
        # self.agent_config.hyperparameters = self.agent_config.hyperparameters["AGENT"]
        # self.agent = DDQN(self.agent_config)
        self.actions_seen = []

        self.min_episode_score_seen = float("inf")

    def save_experience(self, memory=None, experience=None):
        super().save_experience(memory, experience)
        self.actions_seen.append(self.action)
        if self.done:
            self.actions_seen.append("DONE")

    def run_n_episodes(self, num_episodes=None, show_whether_achieved_goal=True, save_and_print_results=True):

        if num_episodes is None: num_episodes = self.config.num_episodes_to_run
        start = time.time()
        while self.episode_number < num_episodes:
            self.reset_game()
            self.step()
            if save_and_print_results: self.save_and_print_result()

            print(self.progress_has_been_made())
        time_taken = time.time() - start
        if show_whether_achieved_goal: self.show_whether_achieved_goal()
        if self.config.save_model: self.locally_save_policy()


        print("--------")
        print("Actions seen")
        print(self.actions_seen)

        return self.game_full_episode_scores, self.rolling_results, time_taken

    def progress_has_been_made(self):
        self.update_min_episode_score()

        current_rolling_result = self.rolling_results[-1]

        improvement = current_rolling_result - self.min_episode_score_seen

        if len(self.game_full_episode_scores) > self.rolling_score_window:
            return improvement > (self.average_score_required_to_win - self.min_episode_score_seen) * 0.2
        return False


    def update_min_episode_score(self):
        """Updates the minimum episode score we have seen so far"""
        if self.total_episode_score_so_far <= self.min_episode_score_seen:
            self.min_episode_score_seen = self.total_episode_score_so_far

        #
        # agent learns in world RANDOMLY first...
        # agent makes some sort of progress...
        # then we infer grammar
        # change final layer, freeze earlier layer
        # train with frozen earlier layers
        # unfreeze earlier layers

        # train model at same time to predict next state...  so it can warn us if we see something that is unusual and so control should go back to agent...
        # also need to encourage agent to pick longer actions rather than just 1 action at a time each time
