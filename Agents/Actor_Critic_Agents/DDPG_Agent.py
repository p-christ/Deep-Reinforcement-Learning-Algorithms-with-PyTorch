import copy
import torch
from torch import optim
from DQN_Agents.DQN_Agent_With_Fixed_Q_Targets import DQN_Agent_With_Fixed_Q_Targets
from Data_Structures.Replay_Buffer import Replay_Buffer
from Model import Model
from Utilities.OU_Noise import OU_Noise


# should use batch normalisation

# Should add noise to the process...

# actor_loss = -self.critic_local(states, actions_pred).mean()

# actor loss is just minus the  Q value for state and actions
# currently the actor and critic learn from DIFFERNET SAMPLEs of experiences, but maybe should be same?
# the critic takes state and action choice in at layer 1 rather than  concatonating them later in the network


class DDPG_Agent(DQN_Agent_With_Fixed_Q_Targets):
    agent_name = "DDPG"

    def __init__(self, config, agent_name):

        config_for_dqn = copy.copy(config)

        for key in config_for_dqn.hyperparameters["Critic"]:
            config_for_dqn.hyperparameters[key] = config_for_dqn.hyperparameters["Critic"][key]

        DQN_Agent_With_Fixed_Q_Targets.__init__(self, config_for_dqn, agent_name)

        self.ddpg_hyperparameters = config.hyperparameters

        self.critic_local = Model(self.state_size + self.action_size, 1, config.seed, self.ddpg_hyperparameters["Critic"]).to(self.device)
        self.critic_target = Model(self.state_size + self.action_size, 1, config.seed, self.ddpg_hyperparameters["Critic"]).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),lr=self.ddpg_hyperparameters["Critic"]["learning_rate"])

        self.actor_local = Model(self.state_size, self.action_size, config.seed, self.ddpg_hyperparameters["Actor"]).to(self.device)
        self.actor_target = Model(self.state_size, self.action_size, config.seed,self.ddpg_hyperparameters["Actor"]).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),lr=self.ddpg_hyperparameters["Actor"]["learning_rate"])

        self.memory = Replay_Buffer(self.ddpg_hyperparameters["Critic"]["buffer_size"], self.ddpg_hyperparameters["batch_size"], config.seed)

        self.noise = OU_Noise(self.action_size, config.seed)


    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        self.environment.reset_environment()
        self.state = self.environment.get_state()
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False
        self.total_episode_score_so_far = 0
        self.episode_step_number = 0
        self.noise.reset()

    def step(self):
        """Runs a step within a game including a learning step if required"""
        self.pick_and_conduct_action()
        self.update_next_state_reward_done_and_score()

        if self.time_for_critic_to_learn():
            self.critic_learn()

        if self.time_for_actor_to_learn():
            self.actor_learn()

        self.save_experience()
        self.state = self.next_state #this is to set the state for the next iteration

    def pick_action(self):

        state = torch.from_numpy(self.state).float().to(self.device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        action += self.noise.sample()

        return action

    def compute_q_values_for_next_states(self, next_states):
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(torch.cat((next_states, actions_next), 1))
        return Q_targets_next

    def compute_expected_q_values(self, states, actions):
        Q_expected = self.critic_local(torch.cat((states, actions), 1))
        return Q_expected

    def time_for_actor_to_learn(self):
        return  self.enough_experiences_to_learn_from() and self.episode_step_number % self.ddpg_hyperparameters["Actor"]["update_every_n_steps"] == 0

    def actor_learn(self):
        states, _, _, _, _ = self.sample_experiences()  # Sample experiences
        actor_loss = self.calculate_actor_loss(states)
        self.take_actor_optimisation_step(actor_loss)

    def calculate_actor_loss(self, states):
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(torch.cat((states, actions_pred), 1)).mean()
        return actor_loss

    def take_actor_optimisation_step(self, actor_loss):
        if self.done: #we only update the learning rate at end of each episode
            self.update_learning_rate(self.hyperparameters["Actor"]["learning_rate"], self.actor_optimizer)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.soft_update_of_target_network(self.actor_local, self.actor_target, self.ddpg_hyperparameters["Actor"]["tau"])



