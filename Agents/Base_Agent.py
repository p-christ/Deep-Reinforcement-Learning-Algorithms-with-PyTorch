import matplotlib.pyplot as plt
import numpy as np
from Memory_Data_Structures.Replay_Buffer import Replay_Buffer
from abc import ABC, abstractmethod

class Base_Agent(object):    
    
    def __init__(self, environment, seed, hyperparameters, rolling_score_length, average_score_required,
                 agent_name):
        self.environment = environment        
        self.action_size = self.environment.get_action_size()
        self.state_size = self.environment.get_state_size()        
        
        self.hyperparameters = hyperparameters
        self.memory = Replay_Buffer(self.hyperparameters["buffer_size"],
                                    self.hyperparameters["batch_size"], seed)
        
        self.rolling_score_length = rolling_score_length
        self.average_score_required = average_score_required 
        self.reset_game()        
        self.game_scores = []
        self.rolling_results = []
        self.max_rolling_score_seen = float("-inf")
        self.agent_name = agent_name
        self.episode_number = 0

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        self.environment.reset_environment()
        self.state = self.environment.get_state()
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False                
        self.score = 0
        self.step_number = 0

    def run_game_n_times(self, num_episodes_to_run=1, save_model=False):

        for episode in range(num_episodes_to_run):
            self.episode_number += 1
            self.run_game_once()
            self.save_and_print_result()          
            self.reset_game()
            
            if self.max_rolling_score_seen > self.average_score_required: #stop once we achieve required score
                break
        
        self.summarise_results()
        if save_model:
            self.save_model()
        return self.game_scores, self.rolling_results

    def run_game_once(self):
        while not self.done:
                self.step()                
    
    def step(self):
        self.pick_and_conduct_action()
        
        self.update_next_state_reward_done_and_score()
        
        if self.time_to_learn():
            self.learn()
        
        self.save_experience()
        self.state = self.next_state #this is to set the state for the next iteration
        

    def pick_and_conduct_action(self):
        self.action = self.pick_action()   
        self.conduct_action()

    @abstractmethod
    def pick_action(self):
        pass

    def conduct_action(self):        
        self.environment.conduct_action(self.action)
  
        
    def update_next_state_reward_done_and_score(self): 
        self.next_state = self.environment.get_next_state()
        self.reward = self.environment.get_reward()
        self.done = self.environment.get_done()
        self.score += self.environment.get_reward()
        
    def save_experience(self):
        self.memory.add(self.state, self.action, self.reward, self.next_state, self.done)

    def time_to_learn(self):
        return self.right_amount_of_steps_taken() and self.enough_experiences_to_learn_from()
    
    def right_amount_of_steps_taken(self):
        return self.step_number % self.hyperparameters["update_every_n_steps"] == 0
    
    def enough_experiences_to_learn_from(self):
        return len(self.memory) > self.hyperparameters["batch_size"]
           
    @abstractmethod
    def learn(self):
        pass
        
    def sample_experiences(self):        
        experiences = self.memory.sample()        
        states, actions, rewards, next_states, dones = experiences
        return states, actions, rewards, next_states, dones       
        
    def save_and_print_result(self):

        self.save_result()         
        self.print_rolling_result()
               
    def save_result(self):
        self.game_scores.append(self.score)  
        self.rolling_results.append(np.mean(self.game_scores[-1 * self.rolling_score_length:]))
        self.save_max_result_seen()
        
    def save_max_result_seen(self):
        
        if self.rolling_results[-1] > self.max_rolling_score_seen:
            if len(self.rolling_results) > self.rolling_score_length:
                self.max_rolling_score_seen = self.rolling_results[-1]
        
    def print_rolling_result(self):

        print("""Episode {}, Rolling score: {}, Max rolling score seen: {}""".format(len(self.game_scores),
                                                                                     self.rolling_results[-1],
                                                                                     self.max_rolling_score_seen), end="\r", flush=True)
                 
                 
    def summarise_results(self):
        self.show_whether_achieved_goal()                                  
        self.visualise_results()
        

    def visualise_results(self):

        plt.plot(self.rolling_results)
        plt.ylabel('Episode score')
        plt.xlabel('Episode number')
        plt.show()   
        
    def show_whether_achieved_goal(self):
        
        index_achieved_goal = self.achieved_required_score_at_index()
        if index_achieved_goal == -1:
            print("\033[91m" + "\033[1m" + 
                  "{} did not achieve required score \n".format(self.agent_name) +
                  "\033[0m" + "\033[0m")
        else:
            print("\033[92m" + "\033[1m" + 
                  "{} achieved required score at episode {} \n".format(self.agent_name,index_achieved_goal) +
                  "\033[0m" + "\033[0m")
                  
        
    def achieved_required_score_at_index(self):
        for ix, score in enumerate(self.rolling_results):
            if score > self.average_score_required:        
                return ix
        return -1
    
    @abstractmethod
    def save_model(self):
        pass


    
