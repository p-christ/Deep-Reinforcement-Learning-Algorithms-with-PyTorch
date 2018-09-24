from abc import ABC, abstractmethod
from Utilities.Utility_Functions import abstract

@abstract
class Base_Environment(ABC):
            
    @abstractmethod
    def conduct_action(self, action):
        """Must conduct the action and update the parameters:
        1. self.next_state
        2. self.reward
        3. self.done
        It must not return anything"""
        pass
    
    @abstractmethod
    def get_action_size(self):
        """Must return the number (integer) of action choices available to the agent in the game"""
        pass
    
    @abstractmethod
    def get_state_size(self):
        """Must return the number (integer) of state datapoints available to the agent in the game"""
        pass    
    
    @abstractmethod
    def get_state(self):
        """Must return the current state of the game"""
        pass
    
    @abstractmethod
    def get_next_state(self):
        """Must return the current next_state of the game"""
        pass    
    
    @abstractmethod
    def get_reward(self):
        """Must return the latest reward of the game"""
        pass    
    
    @abstractmethod
    def get_done(self):
        """Must return the latest done boolean variable of the game which indicates whether it is the end of an episode
        or not"""
        pass
    
    @abstractmethod
    def reset_environment(self):
        """Must reset the environment and update self.state. Must not return anything"""
        pass
    