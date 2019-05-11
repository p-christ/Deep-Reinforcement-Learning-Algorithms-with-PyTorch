from Base_Agent import Base_Agent
import copy
import time
from DDQN import DDQN
from k_Sequitur import k_Sequitur


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
        self.end_of_episode_symbol = "/"
        self.grammar_calculator = k_Sequitur(k=config.hyperparameters["sequitur_k"], end_of_episode_symbol=self.end_of_episode_symbol)

        # self.full_episode_memory = Replay_Buffer(self.hyperparameters["buffer_size"], self.hyperparameters["batch_size"],
        #                             config.seed)



    def save_experience(self, memory=None, experience=None):
        super().save_experience(memory, experience)
        self.actions_seen.append(self.action)
        if self.done:
            self.actions_seen.append(self.end_of_episode_symbol)
        self.track_episodes_data()

    def run_n_episodes(self, num_episodes=None, show_whether_achieved_goal=True, save_and_print_results=True):

        print("Should change it so action grammar only inferred when playing with NO exploration for few episodes")

        if num_episodes is None: num_episodes = self.config.num_episodes_to_run
        start = time.time()
        while self.episode_number < num_episodes:

            while not self.progress_has_been_made():

                self.reset_game()
                self.step()
                if save_and_print_results: self.save_and_print_result()

                self.store_full_episode_data()

            grammar, all_rules, new_count_symbol = self.grammar_calculator.generate_action_grammar(self.actions_seen)

            print("Grammar ", grammar)
            print("All rules ", all_rules)
            print("New count symbol", new_count_symbol)
            assert 1 == 0

            print(self.progress_has_been_made())
        time_taken = time.time() - start
        if show_whether_achieved_goal: self.show_whether_achieved_goal()
        if self.config.save_model: self.locally_save_policy()


        print("--------")
        print("Actions seen")
        print(self.actions_seen)

        return self.game_full_episode_scores, self.rolling_results, time_taken

    def store_full_episode_data(self):
        """Stores the state, next state, reward, done and action information for the latest full episode"""
        self.memory_shaper.add_episode_experience(self.episode_states, self.episode_next_states, self.episode_rewards,
                                                  self.episode_actions, self.episode_dones)

    def progress_has_been_made(self):

        assert self.average_score_required_to_win < 100000, self.average_score_required_to_win

        if len(self.rolling_results) == 0:
            return False

        self.update_min_episode_score()

        current_rolling_result = self.rolling_results[-1]

        improvement = current_rolling_result - self.min_episode_score_seen

        if len(self.game_full_episode_scores) > 20: #self.rolling_score_window:
            return improvement > (self.average_score_required_to_win - self.min_episode_score_seen) * 0.1
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

class DQN wrapper...