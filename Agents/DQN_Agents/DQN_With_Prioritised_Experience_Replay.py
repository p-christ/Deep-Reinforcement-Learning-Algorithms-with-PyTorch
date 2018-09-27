import torch
import torch.nn.functional as F

from DQN_Agents.DQN_Agent import DQN_Agent
from Prioritised_Replay_Buffer import Prioritised_Replay_Buffer

# TODO change time to learn so that it learns more often if our score far away from goal and vice versa


class DQN_With_Prioritised_Experience_Replay(DQN_Agent):


    def __init__(self, environment, seed, hyperparameters, rolling_score_length,
                 average_score_required, agent_name):
        DQN_Agent.__init__(self, environment=environment,
                            seed=seed, hyperparameters=hyperparameters, rolling_score_length=rolling_score_length,
                            average_score_required=average_score_required, agent_name=agent_name)

        self.memory = Prioritised_Replay_Buffer(self.hyperparameters["buffer_size"],
                                                self.hyperparameters["batch_size"],
                                                seed)

    def learn(self):

        sampled_experiences, importance_sampling_weights = self.memory.sample()
        states, actions, rewards, next_states, dones = sampled_experiences
        loss, td_errors = self.compute_loss_and_td_errors(states, next_states, rewards, actions, dones, importance_sampling_weights)

        self.take_optimisation_step(loss)
        # self.soft_update_of_target_network()  # Update the target network
        self.memory.update_td_errors(td_errors.squeeze(1))

    def save_experience(self):
        """Saves the latest experience including the td_error"""
        max_td_error_in_experiences = self.memory.give_max_td_error() + 1e-9
        self.memory.add_experience(max_td_error_in_experiences, self.state, self.action, self.reward, self.next_state, self.done)

    def compute_loss_and_td_errors(self, states, next_states, rewards, actions, dones, importance_sampling_weights):

        Q_targets = self.compute_q_targets(next_states, rewards, dones)
        Q_expected = self.compute_expected_q_values(states, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        # loss = loss * importance_sampling_weights
        # loss = torch.mean(loss)

        td_errors = Q_targets.data.cpu().numpy() - Q_expected.data.cpu().numpy()

        return loss, td_errors

    # def compute_q_targets_and_q_expected(self, states, next_states, rewards, actions, dones):
    #     Q_targets = self.compute_q_targets(next_states, rewards, dones) # dont calculate grads
    #     Q_expected = self.compute_expected_q_values(states, actions) #calculate grads
    #
    #     return Q_targets, Q_expected
    #
    # def compute_q_values_for_next_states(self, next_states):
    #     # self.qnetwork_local.eval()  # puts network in evaluation mode
    #     # self.qnetwork_target.eval()  # puts network in evaluation mode
    #     # with torch.no_grad():
    #     max_action_indexes = self.qnetwork_local(next_states).detach().argmax(1)
    #     Q_targets_next = self.qnetwork_target(next_states).gather(1, max_action_indexes.unsqueeze(1))
    #     # self.qnetwork_local.train()  # puts network back in training mode
    #     # self.qnetwork_target.train()  # puts network back in training mode
    #
    #     return Q_targets_next
    #
    # def compute_td_errors(self, Q_targets, Q_expected):
    #     return Q_targets.data.cpu().numpy() - Q_expected.data.cpu().numpy()
    #
    # def compute_td_errors_times_importance_sampling_weights(self, td_errors, importance_sampling_weights):
    #     importance_sampling_weights = torch.tensor(importance_sampling_weights, requires_grad=False)
    #     td_errors = torch.tensor(td_errors, requires_grad=False)
    #
    #     td_errors_times_importance_weights = importance_sampling_weights.view(-1, 1).float() * td_errors.float()
    #
    #     return td_errors_times_importance_weights





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
        