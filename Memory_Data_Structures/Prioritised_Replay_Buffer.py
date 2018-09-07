from Replay_Buffer import Replay_Buffer

from collections import namedtuple, deque
import random
import torch
import numpy as np

class Prioritised_Replay_Buffer(Replay_Buffer):
    
    def __init__(self, action_size, buffer_size, batch_size, seed, alpha, incremental_priority):
        
        Replay_Buffer.__init__(self, action_size, buffer_size, batch_size, seed)
        
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "td_error"])
        
        self.indices_to_update_td_error_for = None
        self.alpha = alpha
        self.incremental_priority = incremental_priority
        
    def add(self, state, action, reward, next_state, done, td_error):
        
        experience = self.experience(state, action, reward, next_state, done, td_error)
        self.memory.append(experience)
        
    def pick_experiences(self):
        
        probabilities = self.create_probabilities()
        indices = self.pick_indices(probabilities)
        experiences = [self.memory[ix] for ix in indices]

        self.note_indices_to_update_td_error_for(indices)
        
        return experiences
    
    def create_probabilities(self):
        
        td_errors = self.create_list_of_adapted_td_errors()
        total_td_error = sum(td_errors)
        probabilities = td_errors / total_td_error        
        return probabilities
    
    def create_list_of_adapted_td_errors(self):
        
        td_errors = [(abs(experience.td_error) + self.incremental_priority)**self.alpha for experience in list(self.memory)]
        return td_errors

    def pick_indices(self, probabilities):        
        num_experiences = len(probabilities)        
        indices = np.random.choice(num_experiences, self.batch_size, p=probabilities)
        return indices
    
    def note_indices_to_update_td_error_for(self, indices):
        self.indices_to_update_td_error_for = indices                
    
    
    def update_td_errors(self, td_errors):

        for update_list_index, memory_index in enumerate(self.indices_to_update_td_error_for):            
            experience = self.memory[memory_index]
            
            self.memory[memory_index] = self.experience(experience.state, experience.action, experience.reward,
                                                        experience.next_state, experience.done, td_errors[update_list_index])
            
        
        
        
        
        