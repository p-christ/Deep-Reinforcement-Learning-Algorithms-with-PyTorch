from torch import optim

from Base_Agent import Base_Agent
from Data_Structures.Replay_Buffer import Replay_Buffer
from Model import Model


# should use batch normalisation


# actor_loss = -self.critic_local(states, actions_pred).mean()

# actor loss is just minus the  Q value for state and actions

class DDPG_Agent(Base_Agent):

    def __init__(self, config, agent_name):
        Base_Agent.__init__(self, config, agent_name)

        self.critic_local = Model(self.state_size, 1, config.seed, self.hyperparameters["Critic"]).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),lr=self.hyperparameters["Critic"]["learning_rate"])

        self.actor_local = Model(self.state_size, self.action_size, config.seed, self.hyperparameters["Actor"]).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),lr=self.hyperparameters["Actor"]["learning_rate"])

        self.critic_target = Model(self.state_size, 1, config.seed, self.hyperparameters["Critic"]).to(self.device)
        self.actor_target = Model(self.state_size, self.action_size, config.seed, self.hyperparameters["Actor"]).to(self.device)

        self.memory = Replay_Buffer(self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"], config.seed)

    def step(self):
        """Runs a step within a game including a learning step if required"""
        self.pick_and_conduct_action()
        self.update_next_state_reward_done_and_score()

        if self.time_for_critic_to_learn():
            self.critic_learn()
            self.soft_update_of_critic_target()


        if self.time_for_actor_to_learn():
            self.actor_learn()
            self.soft_update_of_actor_target()

        self.save_experience()
        self.state = self.next_state #this is to set the state for the next iteration


    def pick_and_conduct_action(self):
        action = self.pick_action()



