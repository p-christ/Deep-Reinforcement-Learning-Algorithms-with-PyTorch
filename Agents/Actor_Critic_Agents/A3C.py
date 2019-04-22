import torch
from torch import multiprocessing

from Base_Agent import Base_Agent


class A3C(Base_Agent):
    pass



class Actor_Critic_Worker(multiprocessing.Process):
    rank, args, shared_model, counter, lock, optimizer = None

    def __init__(self, worker_num, environment, shared_model, counter, lock, optimizer, config):
        super(Actor_Critic_Worker, self).__init__()

        self.environment = environment
        self.config = config
        self.set_seeds(worker_num)

        self.environment = agent.environment
        self.agent = agent
        self.exploration = exploration
        self.actor_critic = agent.actor_critic
        self.actor_critic_optimizer = agent.actor_critic_optimizer
        self.hyperparameters = agent.hyperparameters
        self.queue = queue

    def set_seeds(self, worker_num):
        """Sets random seeds for this worker"""
        torch.manual_seed(self.config.seed + worker_num)
        self.environment.seed(self.config.seed + worker_num)

    def run(self):
        self.state = self.reset_game_for_worker()
        done = False
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_action_probabilities = []
        while not done:
            action, action_log_prob = self.pick_action() # self.actor_critic, state, self.epsilon_exploration)
            self.environment.conduct_action(action)
            next_state = self.environment.get_next_state()
            reward = self.environment.get_reward()
            done = self.environment.get_done()
            self.episode_states.append(state)
            self.episode_actions.append(action)
            self.episode_rewards.append(reward)
            self.episode_log_action_probabilities.append(action_log_prob)
            state = next_state

        total_loss = self.calculate_total_loss() #episode_states, episode_rewards, episode_log_action_probabilities)
        gradients = self.calculate_gradients(total_loss)

        self.queue.put((gradients, self.episode_rewards, self.episode_states, self.episode_actions))

    def reset_game_for_worker(self):
        """Resets the game environment so it is ready to play a new episode"""
        seed = random.randint(0, sys.maxsize)
        torch.manual_seed(seed)  # Need to do this otherwise each worker generates same experience
        state = self.env.reset()
        if self.agent.action_types == "CONTINUOUS": self.noise.reset()
        return state

    def pick_action(self): #, policy, state, epsilon_exploration=None):
        """Picks an action using the policy"""
        state = torch.from_numpy(self.state).float().unsqueeze(0)
        actor_output = self.actor_critic.forward(state)
        actor_output = actor_output[:, list(range(self.agent.action_size))] #we only use first set of columns to decide action, last column is state-value
        # print("Actor outputs should sum to 1 ", actor_output)
        action_distribution = create_actor_distribution(self.agent.action_types, actor_output, self.agent.action_size)
        action = action_distribution.sample().cpu().numpy()
        if self.agent.action_types == "CONTINUOUS": action += self.noise.sample()
        if self.agent.action_types == "DISCRETE":
            if random.random() <= self.exploration:
                action = random.randint(0, self.agent.action_size - 1)
                action = np.array([action])
        action_log_prob = self.calculate_log_action_probability(action, action_distribution)
        return action, action_log_prob

    def calculate_log_action_probability(self, actions, action_distribution):
        """Calculates the log probability of the chosen action"""
        policy_distribution_log_prob = action_distribution.log_prob(torch.Tensor(actions))
        return policy_distribution_log_prob

    def calculate_total_loss(self): #, episode_states, episode_rewards, episode_log_action_probabilities):
        """Calculates the actor loss + critic loss"""
        discounted_returns = self.calculate_discounted_returns()
        if self.hyperparameters["normalise_rewards"]:
            discounted_returns = self.normalise_discounted_returns(discounted_returns)
        critic_loss, advantages = self.calculate_critic_loss_and_advantages(discounted_returns)
        actor_loss = self.calculate_actor_loss(advantages)
        total_loss = actor_loss + critic_loss
        return total_loss

    def calculate_discounted_returns(self):#, states, rewards):
        """Calculates the cumulative discounted return for an episode which we will then use in a learning iteration"""
        discounted_returns = [0]
        for ix in range(len(self.episode_states)):
            return_value = self.episode_rewards[-(ix + 1)] + self.hyperparameters["discount_rate"]*discounted_returns[-1]
            discounted_returns.append(return_value)
        discounted_returns = discounted_returns[1:]
        discounted_returns = discounted_returns[::-1]
        return discounted_returns

    def normalise_discounted_returns(self, discounted_returns):
        """Normalises the discounted returns by dividing by mean and std of returns that episode"""
        mean = np.mean(discounted_returns)
        std = np.std(discounted_returns)
        discounted_returns -= mean
        discounted_returns /= std
        return discounted_returns

    def calculate_critic_loss_and_advantages(self, all_discounted_returns):
        """Calculates the critic's loss and the advantages"""
        states = torch.Tensor(self.episode_states)
        critic_values = self.actor_critic(states)[:, -1]

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

    def calculate_gradients(self, total_loss):
        """Calculates gradients for the worker"""
        self.actor_critic_optimizer.zero_grad()
        total_loss.backward()
        gradients = [param.grad for param in list(self.actor_critic.parameters())]
        return gradients
