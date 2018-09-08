"""This agent is TBD and not finished yet"""

from Agents.DQN_Agents.DQN_Agent import DQN_Agent
from Memory_Data_Structures.Prioritised_Replay_Buffer import Prioritised_Replay_Buffer
import torch
import numpy as np
import torch.nn.functional as F

from Prioritised_Replay_Buffer_Rank_Prioritisation import Prioritised_Replay_Buffer_Rank_Prioritisation



class DQN_With_Prioritised_Experience_Replay(DQN_Agent):
    
    
    def __init__(self, environment, seed, hyperparameters, rolling_score_length, 
                 average_score_required, agent_name, rank_prio):
        DQN_Agent.__init__(self, environment=environment,
                            seed=seed, hyperparameters=hyperparameters, rolling_score_length=rolling_score_length,
                            average_score_required=average_score_required, agent_name=agent_name)
       
        # We have to use a slightly different data structure to enforce prioritised sampling

        if rank_prio:
            self.memory = Prioritised_Replay_Buffer_Rank_Prioritisation(self.hyperparameters["buffer_size"],
                                                                    self.hyperparameters["batch_size"])

            print("rank prioritisation")
        #
        # if list_data_structure:
        #
        #     self.memory = Prioritised_Replay_Buffer_As_List(self.hyperparameters["buffer_size"],
        #                             self.hyperparameters["batch_size"], seed, alpha=self.hyperparameters["alpha"],
        #                                        incremental_priority=self.hyperparameters["incremental_priority"])
        #
        else:

            self.memory = Prioritised_Replay_Buffer(self.hyperparameters["buffer_size"],
                                        self.hyperparameters["batch_size"], seed, alpha=self.hyperparameters["alpha"],
                                                   incremental_priority=self.hyperparameters["incremental_priority"])

            print("no rank prioritisation")

        self.rank_prio = rank_prio


    
    def save_experience(self):
        """Saves the latest experience including the td_error"""
        self.memory.add(self.state, self.action, self.reward, self.next_state, self.done, self.td_error)

    
    def step(self):
        """Picks and conducts an action, learns and then saves the experience"""
        self.pick_and_conduct_action()
        
        self.update_next_state_reward_done_and_score()
        self.calculate_td_error()
        
        if self.time_to_learn():
            self.learn()
        
        self.save_experience()
        self.state = self.next_state #this is to set the state for the next iteration                
        
    def calculate_td_error(self):
        
        Q_targets = self.calculate_Q_targets_for_td_error_calculation()
        Q_expected = self.calculate_Q_expected_for_td_error_calculation()
        self.td_error = Q_targets - Q_expected
        
                        
        
    def calculate_Q_targets_for_td_error_calculation(self):
        
        next_state = torch.from_numpy(self.next_state).float().unsqueeze(0).to(self.device) 
        
        self.qnetwork_local.eval() #puts network in evaluation mode
        with torch.no_grad():
            Q_targets_next = self.qnetwork_local(next_state).detach().max(1)[0].unsqueeze(1)                        
        self.qnetwork_local.train() #puts network back in training mode    

        Q_targets_next = Q_targets_next.data[0][0].cpu().numpy()
        
        Q_targets = self.reward + (self.hyperparameters["gamma"] * Q_targets_next * (1 - self.done))
        
        return Q_targets
        
    def calculate_Q_expected_for_td_error_calculation(self):
        
        state = torch.from_numpy(self.state).float().unsqueeze(0).to(self.device)
        action = torch.from_numpy(np.vstack([action_val for action_val in [self.action]])).float().to(self.device)
        
        self.qnetwork_local.eval() #puts network in evaluation mode
        with torch.no_grad():
            Q_expected = self.qnetwork_local(state).gather(1, action.long())
        self.qnetwork_local.train() #puts network back in training mode           
        
        Q_expected = Q_expected.data[0][0].cpu().numpy()
        
        return Q_expected
        

    def learn(self):        
        if self.time_to_learn():
            states, actions, rewards, next_states, dones = self.sample_experiences() #Sample experiences                        
            loss, td_errors = self.compute_loss(states, next_states, rewards, actions, dones) #Compute the loss            

            if self.rank_prio:
                self.memory.add_batch(states, actions, rewards, next_states, dones, td_errors)

            else:
                self.memory.update_td_errors(td_errors) # update the td errors for those observsations...

            
            self.take_optimisation_step(loss) #Take an optimisation step            
    
    def compute_loss(self, states, next_states, rewards, actions, dones):
        
        Q_targets = self.compute_q_targets(next_states, rewards, dones)
        Q_expected = self.compute_expected_q_values(states, actions) 
        
        td_errors = Q_targets.data.cpu().numpy() - Q_expected.data.cpu().numpy()
        
        loss = F.mse_loss(Q_expected, Q_targets)
        
        return loss, td_errors[:, 0]
        
        
        