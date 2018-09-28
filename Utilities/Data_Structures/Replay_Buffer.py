from collections import namedtuple, deque
import random
import torch
import numpy as np



class Replay_Buffer(object):
    
    def __init__(self, buffer_size, batch_size, seed):

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def add_experience(self, state, action, reward, next_state, done):
        
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)
   
    def sample(self):        
        experiences = self.pick_experiences()
        states, actions, rewards, next_states, dones = self.separate_out_data_types(experiences)        
        return states, actions, rewards, next_states, dones
            
    def separate_out_data_types(self, experiences):
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([int(e.done) for e in experiences if e is not None])).float().to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def pick_experiences(self):        
        return random.sample(self.memory, k=self.batch_size)        

    def __len__(self):
        
        return len(self.memory)