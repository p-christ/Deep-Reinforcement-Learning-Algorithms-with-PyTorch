import copy

import torch

from Base_Agent import Base_Agent
from Linear_Model import Linear_Model
from NN import NN
import numpy as np



class Hill_Climbing_Agent(Base_Agent):

    def __init__(self, environment, seed, hyperparameters, rolling_score_length, average_score_required,
                 agent_name):

        self.episode_reward = 0
        self.last_episode_reward = 0

        # np.random.seed(0)
        #
        # environment.game_environment.seed(0)



        Base_Agent.__init__(self, environment=environment,
                            seed=seed, hyperparameters=hyperparameters, rolling_score_length=rolling_score_length,
                            average_score_required=average_score_required, agent_name=agent_name)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.latest_network = NN(self.state_size, self.action_size, seed, hyperparameters).to(self.device)
        # self.previous_network = NN(self.state_size, self.action_size, seed, hyperparameters).to(self.device)
        #

        #
        # self.policy = NN(self.state_size, self.action_size, seed, hyperparameters).to(self.device)

        if self.hyperparameters["policy_network_type"] == "Linear":
            self.policy = Linear_Model(self.state_size, self.action_size)
            self.best_weights_seen = self.policy.weights


        # self.previous_policy = Linear_Model(self.state_size, self.action_size)

        self.best_episode_score_seen = float("-inf")


        # self.best_weights_seen = self.policy.state_dict()

        self.noise_scale = 1e-2



    def pick_action(self):

        # values = self.policy.forward(self.state)
        # action = np.argmax(values)

        # action = self.policy.act(self.state)
        #
        policy_values = self.policy.forward(self.state)
        action = np.argmax(policy_values)

        return action
    #
    #

    # def pick_action(self):
    #
    #     state = torch.from_numpy(self.state).float().unsqueeze(0).to(self.device)  # gets state in format ready for network
    #
    #     self.policy.eval()  # puts network in evaluation mode
    #     with torch.no_grad():
    #         action_values = self.policy(state)
    #     self.policy.train()  # puts network back in training mode
    #
    #     action = self.pick_max_action(action_values)
    #
    #     return action
    #
    # def pick_max_action(self, action_values):
    #     return np.argmax(action_values.cpu().data.numpy())


    # def learn(self):
    #
    #     if self.done:
    #         if self.episode_reward >= self.best_episode_score_seen:
    #
    #             self.best_episode_score_seen = self.episode_reward
    #             self.best_weights_seen = copy.deepcopy(self.policy.state_dict())
    #             noise_scale = max(1e-3, self.noise_scale / 2)
    #
    #         else:
    #
    #             noise_scale = min(2, self.noise_scale * 2)
    #             self.policy.load_state_dict(self.best_weights_seen)
    #
    #
    #         for key in self.policy.state_dict().keys():
    #             self.policy.state_dict()[key] = self.perturb_tensor(self.policy.state_dict()[key], noise_scale)
    #
    #         self.policy.load_state_dict(self.policy.state_dict())

    #
    def learn(self):

        if self.done:

            if self.episode_reward >= self.best_episode_score_seen:

                self.best_episode_score_seen = self.episode_reward
                self.best_weights_seen = self.policy.weights
                noise_scale = max(1e-3, self.noise_scale / 2)
                self.policy.weights += noise_scale * np.random.rand(*self.policy.weights.shape)

                #
                # self.last_round_found_better_weights = True

                # self.previous_network = self.latest_network
            else:
                # print("Old network better: {} vs. {}".format(self.episode_reward, self.last_episode_reward))
                # self.last_round_found_better_weights = False

                noise_scale = min(2, self.noise_scale * 2)
                self.policy.weights = self.best_weights_seen + noise_scale * np.random.rand(*self.policy.weights.shape)
    #
            # self.previous_policy.weights = self.best_weights_seen
            # self.last_round_found_better_weights = True


        # self.make_latest_network_a_perturbed_version_of_previous_network()
        # self.make_latest_network_a_perturbed_version_of_previous_policy()


    #   WAS NEVER USING THIS ONE
    # def make_latest_network_a_perturbed_version_of_previous_policy(self):
    #     if self.last_round_found_better_weights:
    #         self.noise_scale = self.noise_scale / 2.0
    #
    #             # max(1e-3, self.noise_scale / 2)
    #     else:
    #         self.noise_scale = self.noise_scale * 1.1
    #             # min(2, self.noise_scale * 2)
    #
    #     # print("Latest weights")
    #     # print(self.latest_policy.weights)
    #
    #     self.latest_policy.weights = self.previous_policy.weights + \
    #                                  self.noise_scale * (np.random.rand(*self.latest_policy.weights.shape) - 0.5)

        # print("New Latest weights")
        # print(self.latest_policy.weights)





    # def make_latest_network_a_perturbed_version_of_previous_network(self):
    #
    #     state_dict = self.previous_network.state_dict()
    #     state_dict = copy.deepcopy(state_dict)
    #
    #     keys = state_dict.keys()
    #
    #     for key in keys:
    #
    #         state_dict[key] = self.perturb_tensor(state_dict[key])
    #
    #     self.latest_network.load_state_dict(state_dict)
    #
    #
    # def perturb_tensor(self, tensor, noise_scale):
    #
    #     random_tensor = torch.rand(*tensor.shape) - 0.5
    #
    #     tensor += noise_scale * random_tensor
    #
    #     return tensor

    #             noise_scale = max(1e-3, noise_scale / 2)
    #             policy.w += noise_scale * np.random.rand(*policy.w.shape)
    #         else:  # did not find better weights
    #             noise_scale = min(2, noise_scale * 2)
    #             policy.w = best_w + noise_scale * np.random.rand(*policy.w.shape)

    # def save_model(self):
    #     torch.save(self.qnetwork_local.state_dict(), "Models/{}_local_network.pt".format(self.agent_name))

    def update_next_state_reward_done_and_score(self):
        self.next_state = self.environment.get_next_state()
        self.reward = self.environment.get_reward()
        self.done = self.environment.get_done()
        self.score += self.environment.get_reward()

        self.episode_reward += self.reward

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        self.environment.reset_environment()
        self.state = self.environment.get_state()
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False
        self.score = 0
        self.step_number = 0

        self.last_episode_reward = self.episode_reward

        self.episode_reward = 0



# def hill_climbing(n_episodes=1000, max_t=1000, gamma=1.0, print_every=100, noise_scale=1e-2):
#     """Implementation of hill climbing with adaptive noise scaling.
#
#     Params
#     ======
#         n_episodes (int): maximum number of training episodes
#         max_t (int): maximum number of timesteps per episode
#         gamma (float): discount rate
#         print_every (int): how often to print average score (over last 100 episodes)
#         noise_scale (float): standard deviation of additive noise
#     """
#     scores_deque = deque(maxlen=100)
#     scores = []
#     best_R = -np.Inf
#     best_w = policy.w
#     for i_episode in range(1, n_episodes + 1):
#         rewards = []
#         state = env.reset()
#         for t in range(max_t):
#             action = policy.act(state)
#             state, reward, done, _ = env.step(action)
#             rewards.append(reward)
#             if done:
#                 break
#         scores_deque.append(sum(rewards))
#         scores.append(sum(rewards))
#
#         discounts = [gamma ** i for i in range(len(rewards) + 1)]
#         R = sum([a * b for a, b in zip(discounts, rewards)])
#
#         if R >= best_R:  # found better weights
#             best_R = R
#             best_w = policy.w
#             noise_scale = max(1e-3, noise_scale / 2)
#             policy.w += noise_scale * np.random.rand(*policy.w.shape)
#         else:  # did not find better weights
#             noise_scale = min(2, noise_scale * 2)
#             policy.w = best_w + noise_scale * np.random.rand(*policy.w.shape)
#
#         if i_episode % print_every == 0:
#             print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
#         if np.mean(scores_deque) >= 195.0:
#             print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
#                                                                                        np.mean(scores_deque)))
#             policy.w = best_w
#             break
#
#     return scores