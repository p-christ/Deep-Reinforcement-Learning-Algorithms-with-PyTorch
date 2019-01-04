from abc import ABC, abstractmethod

"""An abstract base environment which enforces what methods an environment needs in order to work with the base agent
 and therefore all the agents in the Agents folder. 
 
i.e. for any game, if you create an environment class to represent it that extends this class and implements the below
abstract methods then ALL the agents will be able to play the game"""

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

    @abstractmethod
    def get_max_steps_per_episode(self):
        """Must return the max number of steps per episode"""
        pass

    @abstractmethod
    def get_action_types(self):
        """Must return 'DISCRETE' if actions are discrete and 'CONTINUOUS' if they are continuous"""
        pass

    @abstractmethod
    def get_score_to_win(self):
        """Must return the numerical score required to 'win' the game"""
        pass

    @abstractmethod
    def get_rolling_period_to_calculate_score_over(self):
        """Must return the period of episodes over which the average score needs to be above the score to win in
        order to win"""
        pass
