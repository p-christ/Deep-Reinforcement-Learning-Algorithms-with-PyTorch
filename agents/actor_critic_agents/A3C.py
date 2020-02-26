import copy
import random
import time
import numpy as np
import torch
from torch import multiprocessing
from torch.multiprocessing import Queue
from torch.optim import Adam
from agents.Base_Agent import Base_Agent
from utilities.Utility_Functions import create_actor_distribution, SharedAdam

class A3C(Base_Agent):
    """Actor critic A3C algorithm from deepmind paper https://arxiv.org/pdf/1602.01783.pdf"""
    agent_name = "A3C"
    def __init__(self, config):
        super(A3C, self).__init__(config)
        self.num_processes = multiprocessing.cpu_count()
        self.worker_processes = max(1, self.num_processes - 2)
        self.actor_critic = self.create_NN(input_dim=self.state_size, output_dim=[self.action_size, 1])
        self.actor_critic_optimizer = SharedAdam(self.actor_critic.parameters(), lr=self.hyperparameters["learning_rate"], eps=1e-4)

    def run_n_episodes(self):
        """Runs game to completion n times and then summarises results and saves model (if asked to)"""
        start = time.time()
        results_queue = Queue()
        gradient_updates_queue = Queue()
        episode_number = multiprocessing.Value('i', 0)
        self.optimizer_lock = multiprocessing.Lock()
        episodes_per_process = int(self.config.num_episodes_to_run / self.worker_processes) + 1
        processes = []
        self.actor_critic.share_memory()
        self.actor_critic_optimizer.share_memory()

        optimizer_worker = multiprocessing.Process(target=self.update_shared_model, args=(gradient_updates_queue,))
        optimizer_worker.start()

        for process_num in range(self.worker_processes):
            worker = Actor_Critic_Worker(process_num, copy.deepcopy(self.environment), self.actor_critic, episode_number, self.optimizer_lock,
                                    self.actor_critic_optimizer, self.config, episodes_per_process,
                                    self.hyperparameters["epsilon_decay_rate_denominator"],
                                    self.action_size, self.action_types,
                                    results_queue, copy.deepcopy(self.actor_critic), gradient_updates_queue)
            worker.start()
            processes.append(worker)
        self.print_results(episode_number, results_queue)
        for worker in processes:
            worker.join()
        optimizer_worker.kill()

        time_taken = time.time() - start
        return self.game_full_episode_scores, self.rolling_results, time_taken

    def print_results(self, episode_number, results_queue):
        """Worker that prints out results as they get put into a queue"""
        while True:
            with episode_number.get_lock():
                carry_on = episode_number.value < self.config.num_episodes_to_run
            if carry_on:
                if not results_queue.empty():
                    self.total_episode_score_so_far = results_queue.get()
                    self.save_and_print_result()
            else: break

    def update_shared_model(self, gradient_updates_queue):
        """Worker that updates the shared model with gradients as they get put into the queue"""
        while True:
            gradients = gradient_updates_queue.get()
            with self.optimizer_lock:
                self.actor_critic_optimizer.zero_grad()
                for grads, params in zip(gradients, self.actor_critic.parameters()):
                    params._grad = grads  # maybe need to do grads.clone()
                self.actor_critic_optimizer.step()

class Actor_Critic_Worker(torch.multiprocessing.Process):
    """Actor critic worker that will play the game for the designated number of episodes """
    def __init__(self, worker_num, environment, shared_model, counter, optimizer_lock, shared_optimizer,
                 config, episodes_to_run, epsilon_decay_denominator, action_size, action_types, results_queue,
                 local_model, gradient_updates_queue):
        super(Actor_Critic_Worker, self).__init__()
        self.environment = environment
        self.config = config
        self.worker_num = worker_num

        self.gradient_clipping_norm = self.config.hyperparameters["gradient_clipping_norm"]
        self.discount_rate = self.config.hyperparameters["discount_rate"]
        self.normalise_rewards = self.config.hyperparameters["normalise_rewards"]

        self.action_size = action_size
        self.set_seeds(self.worker_num)
        self.shared_model = shared_model
        self.local_model = local_model
        self.local_optimizer = Adam(self.local_model.parameters(), lr=0.0, eps=1e-4)
        self.counter = counter
        self.optimizer_lock = optimizer_lock
        self.shared_optimizer = shared_optimizer
        self.episodes_to_run = episodes_to_run
        self.epsilon_decay_denominator = epsilon_decay_denominator
        self.exploration_worker_difference = self.config.hyperparameters["exploration_worker_difference"]
        self.action_types = action_types
        self.results_queue = results_queue
        self.episode_number = 0

        self.gradient_updates_queue = gradient_updates_queue

    def set_seeds(self, worker_num):
        """Sets random seeds for this worker"""
        torch.manual_seed(self.config.seed + worker_num)
        self.environment.seed(self.config.seed + worker_num)

    def run(self):
        """Starts the worker"""
        torch.set_num_threads(1)
        for ep_ix in range(self.episodes_to_run):
            with self.optimizer_lock:
                Base_Agent.copy_model_over(self.shared_model, self.local_model)
            epsilon_exploration = self.calculate_new_exploration()
            state = self.reset_game_for_worker()
            done = False
            self.episode_states = []
            self.episode_actions = []
            self.episode_rewards = []
            self.episode_log_action_probabilities = []
            self.critic_outputs = []

            while not done:
                action, action_log_prob, critic_outputs = self.pick_action_and_get_critic_values(self.local_model, state, epsilon_exploration)
                next_state, reward, done, _ =  self.environment.step(action)
                self.episode_states.append(state)
                self.episode_actions.append(action)
                self.episode_rewards.append(reward)
                self.episode_log_action_probabilities.append(action_log_prob)
                self.critic_outputs.append(critic_outputs)
                state = next_state

            total_loss = self.calculate_total_loss()
            self.put_gradients_in_queue(total_loss)
            self.episode_number += 1
            with self.counter.get_lock():
                self.counter.value += 1
                self.results_queue.put(np.sum(self.episode_rewards))

    def calculate_new_exploration(self):
        """Calculates the new exploration parameter epsilon. It picks a random point within 3X above and below the
        current epsilon"""
        with self.counter.get_lock():
            epsilon = 1.0 / (1.0 + (self.counter.value / self.epsilon_decay_denominator))
        epsilon = max(0.0, random.uniform(epsilon / self.exploration_worker_difference, epsilon * self.exploration_worker_difference))
        return epsilon

    def reset_game_for_worker(self):
        """Resets the game environment so it is ready to play a new episode"""
        state = self.environment.reset()
        if self.action_types == "CONTINUOUS": self.noise.reset()
        return state

    def pick_action_and_get_critic_values(self, policy, state, epsilon_exploration=None):
        """Picks an action using the policy"""
        state = torch.from_numpy(state).float().unsqueeze(0)
        model_output = policy.forward(state)
        actor_output = model_output[:, list(range(self.action_size))] #we only use first set of columns to decide action, last column is state-value
        critic_output = model_output[:, -1]
        action_distribution = create_actor_distribution(self.action_types, actor_output, self.action_size)
        action = action_distribution.sample().cpu().numpy()
        if self.action_types == "CONTINUOUS": action += self.noise.sample()
        if self.action_types == "DISCRETE":
            if random.random() <= epsilon_exploration:
                action = random.randint(0, self.action_size - 1)
            else:
                action = action[0]
        action_log_prob = self.calculate_log_action_probability(action, action_distribution)
        return action, action_log_prob, critic_output

    def calculate_log_action_probability(self, actions, action_distribution):
        """Calculates the log probability of the chosen action"""
        policy_distribution_log_prob = action_distribution.log_prob(torch.Tensor([actions]))
        return policy_distribution_log_prob

    def calculate_total_loss(self):
        """Calculates the actor loss + critic loss"""
        discounted_returns = self.calculate_discounted_returns()
        if self.normalise_rewards:
            discounted_returns = self.normalise_discounted_returns(discounted_returns)
        critic_loss, advantages = self.calculate_critic_loss_and_advantages(discounted_returns)
        actor_loss = self.calculate_actor_loss(advantages)
        total_loss = actor_loss + critic_loss
        return total_loss

    def calculate_discounted_returns(self):
        """Calculates the cumulative discounted return for an episode which we will then use in a learning iteration"""
        discounted_returns = [0]
        for ix in range(len(self.episode_states)):
            return_value = self.episode_rewards[-(ix + 1)] + self.discount_rate*discounted_returns[-1]
            discounted_returns.append(return_value)
        discounted_returns = discounted_returns[1:]
        discounted_returns = discounted_returns[::-1]
        return discounted_returns

    def normalise_discounted_returns(self, discounted_returns):
        """Normalises the discounted returns by dividing by mean and std of returns that episode"""
        mean = np.mean(discounted_returns)
        std = np.std(discounted_returns)
        discounted_returns -= mean
        discounted_returns /= (std + 1e-5)
        return discounted_returns

    def calculate_critic_loss_and_advantages(self, all_discounted_returns):
        """Calculates the critic's loss and the advantages"""
        critic_values = torch.cat(self.critic_outputs)
        advantages = torch.Tensor(all_discounted_returns) - critic_values
        advantages = advantages.detach()
        critic_loss =  (torch.Tensor(all_discounted_returns) - critic_values)**2
        critic_loss = critic_loss.mean()
        return critic_loss, advantages

    def calculate_actor_loss(self, advantages):
        """Calculates the loss for the actor"""
        action_log_probabilities_for_all_episodes = torch.cat(self.episode_log_action_probabilities)
        actor_loss = -1.0 * action_log_probabilities_for_all_episodes * advantages
        actor_loss = actor_loss.mean()
        return actor_loss

    def put_gradients_in_queue(self, total_loss):
        """Puts gradients in a queue for the optimisation process to use to update the shared model"""
        self.local_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), self.gradient_clipping_norm)
        gradients = [param.grad.clone() for param in self.local_model.parameters()]
        self.gradient_updates_queue.put(gradients)

