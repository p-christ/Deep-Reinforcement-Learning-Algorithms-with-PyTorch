"""This agent is TBD and not finished yet"""

from DQN_Agents.DDQN_Agent import DDQN_Agent
import torch
import torch.nn.functional as F

from Prioritised_Replay_Buffer_Segment_Tree_Impl import Prioritised_Replay_Buffer_Segment_Tree_Impl

# TODO change time to learn so that it learns more often if our score far away from goal and vice versa


class DDQN_With_Prioritised_Experience_Replay(DDQN_Agent):


    def __init__(self, environment, seed, hyperparameters, rolling_score_length,
                 average_score_required, agent_name):
        DDQN_Agent.__init__(self, environment=environment,
                            seed=seed, hyperparameters=hyperparameters, rolling_score_length=rolling_score_length,
                            average_score_required=average_score_required, agent_name=agent_name)

        # We have to use a slightly different data structure to enforce prioritised sampling
        self.memory = Prioritised_Replay_Buffer_Segment_Tree_Impl(self.hyperparameters["buffer_size"],
                                                                self.hyperparameters["batch_size"], seed,
                                                                  self.hyperparameters["alpha_prioritised_replay"],
                                                                  self.hyperparameters["beta_prioritised_replay"],
                                                                   self.state_size)



    def save_experience(self):
        """Saves the latest experience including the td_error"""
        self.memory.add(self.state, self.action, self.reward, self.next_state, self.done)


    def step(self):
        """Picks and conducts an action, learns and then saves the experience"""
        self.pick_and_conduct_action()

        self.update_next_state_reward_done_and_score()
        # self.calculate_td_error()

        if self.time_to_learn():
            self.learn()

        self.save_experience()
        self.state = self.next_state #this is to set the state for the next iteration

    def learn(self):
        if self.time_to_learn():

            indexes_to_update_td_error_for, importance_sampling_weights, sampled_experiences = self.memory.sample()

            states, actions, rewards, next_states, dones = sampled_experiences #Sample experiences

            # Q_targets, Q_expected = self.compute_q_targets_and_q_expected(states, next_states, rewards, actions, dones)
            #
            # td_errors = self.compute_td_errors(Q_targets, Q_expected)



            # td_errors_times_importance_weights = self.compute_td_errors_times_importance_sampling_weights(td_errors, importance_sampling_weights)

            importance_sampling_weights = torch.tensor(importance_sampling_weights, requires_grad=False)

            loss, td_errors = self.compute_loss_and_td_errors(states, next_states, rewards, actions, dones)

            loss = loss * importance_sampling_weights.float()
            loss = loss.mean()

            self.memory.update_td_errors(indexes_to_update_td_error_for, td_errors.squeeze(1))

            # self.take_optimisation_step(Q_expected, td_errors_times_importance_weights)

            self.take_optimisation_step(loss)
            self.soft_update_of_target_network()  # Update the target network



    def compute_loss_and_td_errors(self, states, next_states, rewards, actions, dones):

        Q_targets = self.compute_q_targets(next_states, rewards, dones)
        Q_expected = self.compute_expected_q_values(states, actions)
        loss = F.mse_loss(Q_expected, Q_targets)

        td_errors = Q_targets.data.cpu().numpy() - Q_expected.data.cpu().numpy()

        return loss, td_errors

    def compute_q_targets_and_q_expected(self, states, next_states, rewards, actions, dones):
        Q_targets = self.compute_q_targets(next_states, rewards, dones) # dont calculate grads
        Q_expected = self.compute_expected_q_values(states, actions) #calculate grads

        return Q_targets, Q_expected

    def compute_q_values_for_next_states(self, next_states):
        # self.qnetwork_local.eval()  # puts network in evaluation mode
        # self.qnetwork_target.eval()  # puts network in evaluation mode
        # with torch.no_grad():
        max_action_indexes = self.qnetwork_local(next_states).detach().argmax(1)
        Q_targets_next = self.qnetwork_target(next_states).gather(1, max_action_indexes.unsqueeze(1))
        # self.qnetwork_local.train()  # puts network back in training mode
        # self.qnetwork_target.train()  # puts network back in training mode

        return Q_targets_next

    def compute_td_errors(self, Q_targets, Q_expected):
        return Q_targets.data.cpu().numpy() - Q_expected.data.cpu().numpy()

    def compute_td_errors_times_importance_sampling_weights(self, td_errors, importance_sampling_weights):
        importance_sampling_weights = torch.tensor(importance_sampling_weights, requires_grad=False)
        td_errors = torch.tensor(td_errors, requires_grad=False)

        td_errors_times_importance_weights = importance_sampling_weights.view(-1, 1).float() * td_errors.float()

        return td_errors_times_importance_weights





    # def take_optimisation_step(self, Q_expected, td_errors_times_importance_weights):
    #
    #     self.optimizer.zero_grad()  # reset gradients to 0
    #     Q_expected.backward(td_errors_times_importance_weights)  # this calculates the gradients
    #     self.optimizer.step()  # this applies the gradients
    #
    #







    # def compute_loss(self, states, next_states, rewards, actions, dones, importance_sampling_weights):
    #
    #     Q_targets = self.compute_q_targets(next_states, rewards, dones)
    #     Q_expected = self.compute_expected_q_values(states, actions)
    #
    #     td_errors = Q_targets.data.cpu().numpy() - Q_expected.data.cpu().numpy()
    #
    #     importance_sampling_weights = torch.tensor(importance_sampling_weights, requires_grad=False)
    #     # print(importance_sampling_weights)
    #
    #
    #     # make a tensor and don't require gradients
    #
    #     #
    #     # print("GO THROUGH PYTORCH DOCUMENATION TO FIGURE THIS OUT")
    #     # np.array(importance_sampling_weights)
    #     # print("GO THROUGH PYTORCH DOCUMENATION TO FIGURE THIS OUT")
    #     #
    #     # print("And see paper to figure out where to incorporate it really?")
    #     #
    #     #
    #     # print(importance_sampling_weights[0])
    #
    #
    #     loss = F.mse_loss(Q_expected, Q_targets)  #* importance_sampling_weights
    #
    #
    #     loss = torch.mul(loss.float(), importance_sampling_weights.float())
    #     # print(loss)
    #
    #
    #
    #     return loss, td_errors[:, 0]
    #
    #
        