import random
from collections import Counter

import torch
from gym import Wrapper, spaces
from torch import nn, optim
from Base_Agent import Base_Agent
import copy
import time
import numpy as np
from DDQN import DDQN
from HRL.DDQN_HRL import DDQN_Wrapper
from HRL.Grammar_Generator import Grammar_Generator
from Memory_Shaper import Memory_Shaper
from Utility_Functions import flatten_action_id_to_actions
from k_Sequitur import k_Sequitur
import numpy as np
from operator import itemgetter

class HRL(Base_Agent):
    agent_name = "HRL"

    def __init__(self, config):
        super().__init__(config)
        self.grammar_generator = Grammar_Generator(self.hyperparameters["num_top_results_to_use"], self.action_size,
                                                   self.hyperparameters["add_1_macro_action_at_a_time"], self.hyperparameters["use_relative_counts"],
                                                   self.hyperparameters[
                                                       "reduce_macro_action_appearance_cutoff_throughout_training"], self.logger,
                                                   self.hyperparameters["sequitur_k"],
                                                   self.hyperparameters["action_frequency_required_in_top_results"])
        self.rolling_score = self.lowest_possible_episode_score
        self.episodes_to_run_with_no_exploration = self.hyperparameters["episodes_to_run_with_no_exploration"]
        self.action_id_to_action = {k: tuple([k]) for k in range(self.action_size)}
        self.episodes_per_round = self.hyperparameters["episodes_per_round"]
        self.agent = DDQN_Wrapper(config, self.action_id_to_action)

    def run_n_episodes(self, num_episodes=None, show_whether_achieved_goal=True, save_and_print_results=True):
        start = time.time()
        if num_episodes is None: num_episodes = self.config.num_episodes_to_run
        self.episodes_conducted = 0
        while self.episodes_conducted < num_episodes:
            playing_data, self.round_of_macro_actions = self.agent.run_n_episodes(num_episodes=self.episodes_per_round,
                                                                                  episodes_to_run_with_no_exploration=self.episodes_to_run_with_no_exploration)
            self.episodes_conducted += len(playing_data)
            self.action_id_to_action, new_actions_just_added =  self.grammar_generator.generate_new_grammar(playing_data,
                                                                                                           self.action_id_to_action)
            self.agent.update_agent(self.action_id_to_action, new_actions_just_added)
        print("Final episode set actions count: ", Counter(self.round_of_macro_actions))
        time_taken = time.time() - start
        return self.agent.game_full_episode_scores[:num_episodes], self.agent.rolling_results[:num_episodes], time_taken




# Presentation ideas:
# 1) Show that agent at end is using macro actions and not ignoring them
# 2) Show an example final episode using a macro action
# 3) Use 10 random seeds to compare algos
# 4) Compare DDQN with this algo
# 5) Point is this game too simple to benefit from macro actions but it is using them
# 6) Use name Hindsight Macro-Action Experience Replay
# 7) We have extended idea of intrinsic motivation to apply to picking longer macro-actions (rather than exploration)
# 8) having network train on next state prediction task has 2 benefits: 1) lets us know when to abandon macro action & 2) like in UNREAL can speed up training

# TODO train model to predict next state too so that we can use this to figure out when to abandon macro actions
# TODO do grammar updates less often as we go...  increase % improvement required as we go?
# TODO bias grammar inference to be bias towards longer actions
# TODO don't base grammar updates on progress
# TODO do pre-training until loss stops going down quickly?
# TODO Use SAC-Discrete instead of DQN?
# TODO learn grammar of a mixture of best performing episodes when had exploration on vs. not
# TODO higher learning rate when just training final layer?
# TODO write a check that the final layers of network learn properly after we change them
# TODO add mechanism to go backwards if we picked an earlier macro action that then becomes irrelevant?
# TODO change so its the action rules used in most of best performing episodes that get used rather than
#      those that occur the most overall. because a better episode ends faster and so less occurances of actions!!
#      maybe even pick out the fixed X many actions from the top performing episodes no matter how many episodes
# TODO have higher minimum bound for number episodes to retrain on
# TODO try starting with all the 2 step moves as macro actions...  or even starting with random macro actions?
# TODO try having the threshold for how often need see actions come down throughout training?
# TODO fix the tests for memory shaper... very important this works properly
# TODO try just learning off the top 2 episodes? instead of top 10?
# TODO pick actions that are most common in best episodes but also not common in the less well preforming episodes
# TODO could refresh the macro actions we pick later in game
# TODO add option for full random play for some episodes
# TODO try a constant exploration rate
# TODO have DQN agent using an action balanced replay buffer from START not just from 1st grammar iteration?
# TODO try where number of pre-training iterations is fixed and doesnt depend on number of new actions
# TODO try adding 1 action at a time?
# TODO make multi action q-values more self consistent by forcing them to be the first actions q-values + some value
# TODO use a fixed leraning rate when training new actions rather than a learning rate that is lower because we near required score
# TODO add option for using no_grad when combining actions values from different places
# TODO reduce exploration as number of actions increases?
# Having longer actions can replace exploration
# TODO pick macro actions common in best performing episodes but NOT common in worse performing episodes...
# TODO create a toy game that it should specifically be able to solve while other algorithms cannot as easily
# TODO Predict next state. Abandon if not predicted. Reward for getting to unpredicted states (exploration)
# TODO try abandon macro action if:  1) Predict next state model throws big error  2) Next primitive action isn't highest one (by some threshold)
# TODO check if buffer for primitive actions not being filled up by double counting on primitive action moves
# TODO try different size learning rates to learn actions of different length? Also check what LR being used generally
# TODO why not try increasing k in k-sequitur and adding many actions at once... might get different results now
# TODO try increasing rewards for all experiences in those episodes where the overall agent did really well.. related to soft q imitation learning
# TODO is there anyway of making it more end-to-end?
# TODO could use model of world to predict next state and then use that to pre-decide next action and so on until a surprisal happens (is this action grammars though)
# TODO instead could feed in predicted next states into Q network so that it uses them to pick its macro action
# TODO try only keeping the top 50% of experiences in replay buffer and removing the rest vs. only keeping top and bottom 20%
# TODO try with DQN base not DDQN base?
# TODO having network train on next state prediction task has 2 benefits: 1) lets us know when to abandon macro action & 2) like in UNREAL can speed up training
# TODO have a network that predicts next state and abandon macro-action when its hard to predict what next state is, not just if it gets it wrong
# TODO use games where visual input not that important... or download pre-trained DQN
# download a pre-trained DQN to speed things up... for atari
# use diagram of my model
# use comparison table with other columns for papers ...
# include other models that look at hierarchies in different ways...
# discretize actions
# HER with hindsight over the actions
# TODO must leverage macro actions in order to do structured exploration... exploration a v. important benefit of this all...
# TODO use sacred to log all experiment results https://github.com/IDSIA/sacred
# TODO try using RMSProp optimizer instead...
# TODO try just restricting actions to having to pick 2 actions in a row. 0 1 --> 00 10 01 11  see what happens
# TODO 2 stage... first stage whether do habit or not, next stage whether do other action
# TODO brain changes to make it easier / less effort to repeat habits. How can we replicate that idea here?
# TODO human actions result of combination of part of brain that acts to get reward vs. part of brain that does habits
# TODO put a minimum limit on number of instances of a macro action in replay buffer before we add it as a macro action?
# Note that exploration might become worse if you have lots of macro actions using same actions over and over.. e.g. in Taxi they might be less likely
# to pick action 5 if you got loads of macro actions where it doesnt use it. it's like getting stuck in an exploration local min in a way
# TODO have state predictor predict next state, then get q-value from that point after picking next move... so that all we need to learn is
# the q-value of the primitive values really....  fewer things to learn... and then you don't even need a balanced replay buffer or hindsight

