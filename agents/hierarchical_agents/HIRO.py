import copy
import torch
import numpy as np
from gym import Wrapper
from agents.Base_Agent import Base_Agent
from agents.actor_critic_agents.DDPG import DDPG
from agents.Trainer import Trainer


class HIRO(Base_Agent):
    agent_name = "HIRO"

    def __init__(self, config):
        super().__init__(config)
        self.max_sub_policy_timesteps = config.hyperparameters["LOWER_LEVEL"]["max_lower_level_timesteps"]
        self.config.hyperparameters = self.config.hyperparameters

        self.higher_level_state = None #true state of environment
        self.higher_level_next_state = None

        self.higher_level_reward = None
        self.lower_level_reward = None

        self.higher_level_done = False
        self.lower_level_done = False

        self.goal = None

        self.lower_level_state = None #state of environment with goal appended
        self.lower_level_next_state = None

        self.lower_level_agent_config = copy.deepcopy(config)
        self.lower_level_agent_config.hyperparameters = self.lower_level_agent_config.hyperparameters["LOWER_LEVEL"]

        self.lower_level_agent_config.environment = Lower_Level_Agent_Environment_Wrapper(self.environment, self, self.max_sub_policy_timesteps)
        self.lower_level_agent = DDPG(self.lower_level_agent_config)

        self.lower_level_agent.average_score_required_to_win = float("inf")

        self.higher_level_agent_config = copy.deepcopy(config)
        self.higher_level_agent_config.hyperparameters = self.higher_level_agent_config.hyperparameters["HIGHER_LEVEL"]
        self.higher_level_agent_config.environment = Higher_Level_Agent_Environment_Wrapper(self.environment, self)
        self.higher_level_agent = HIRO_Higher_Level_DDPG_Agent(self.higher_level_agent_config, self.lower_level_agent.actor_local)

        self.step_lower_level_states = []
        self.step_lower_level_action_seen = []


    def run_n_episodes(self):
        """Runs game to completion n times and then summarises results and saves model (if asked to)"""
        self.higher_level_agent.run_n_episodes(self.config.num_episodes_to_run)

    @staticmethod
    def goal_transition(state, goal, next_state):
        """Provides updated goal according to the goal transition function in the HIRO paper"""
        return state + goal - next_state

    def save_higher_level_experience(self):
        self.higher_level_agent.step_lower_level_states = self.step_lower_level_states
        self.higher_level_agent.step_lower_level_action_seen = self.step_lower_level_action_seen

class HIRO_Higher_Level_DDPG_Agent(DDPG):
    """Extends DDPG so that it can function as the higher level agent in the HIRO hierarchical RL algorithm. This only involves
    changing how the agent saves experiences and samples them for learning"""

    def __init__(self, config, lower_level_policy):
        super(HIRO_Higher_Level_DDPG_Agent, self).__init__(config)
        self.lower_level_policy = lower_level_policy
        self.number_goal_candidates = config.hyperparameters["number_goal_candidates"]

    def save_experience(self, memory=None, experience=None):
        """Saves the recent experience to the memory buffer. Adapted from normal DDPG so that it saves the sequence of
        states, goals and actions that we saw whilst control was given to the lower level"""
        if memory is None: memory = self.memory
        if experience is None: experience = self.step_lower_level_states, self.step_lower_level_action_seen, self.reward, self.next_state, self.done
        memory.add_experience(*experience)

    def sample_experiences(self):
        experiences = self.memory.produce_action_and_action_info(separate_out_data_types=False)
        assert len(experiences[0].state) == self.hyperparameters["max_lower_level_timesteps"] or experiences[0].done
        assert experiences[0].state[0].shape[0] == self.state_size * 2
        assert len(experiences[0].action) == self.hyperparameters["max_lower_level_timesteps"] or experiences[0].done

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for ix, experience in enumerate(experiences):
            state, action, reward, next_state, done = self.transform_goal_to_one_most_likely_to_have_induced_actions(experience)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        states = torch.from_numpy(np.vstack([state for state in states])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([action for action in actions])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([reward for reward in rewards])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([next_state for next_state in next_states])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([int(done) for done in dones])).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def transform_goal_to_one_most_likely_to_have_induced_actions(self, experience):
        """Transforms the goal in an experience to the goal that would have been most likely to induce the actions chosen
        by the lower level agent in the experience"""
        goal_candidate_state_change = [experience.state[-1][:self.state_size] - experience.state[0][:self.state_size]]
        goal_candidate_actual_goal = [experience.state[0][self.state_size:]]
        goal_candidate_state_change_random_iterations = [np.random.normal(goal_candidate_state_change[0]) for _ in range(self.number_goal_candidates - 2)]
        goal_candidates = goal_candidate_state_change + goal_candidate_actual_goal + goal_candidate_state_change_random_iterations

        max = float("-inf")
        timesteps_in_experience = len(experience.state)

        for goal_ix, goal in enumerate(goal_candidates):
            log_probability_total = 0
            for state_ix in range(timesteps_in_experience):
                state_obs = experience.state[state_ix][:self.state_size]
                action = experience.action[state_ix]
                log_probability= self.log_probability_lower_level_picks_action(state_obs, goal, action)
                log_probability_total += log_probability
                if state_ix != timesteps_in_experience - 1:
                    next_state = experience.state[state_ix+1][:self.state_size]
                    goal = HIRO.goal_transition(state_obs, goal, next_state)
            if log_probability_total >= max:
                max = log_probability_total
                best_goal_ix = goal_ix

        state = experience.state[0][:self.state_size]
        next_state = experience.next_state
        reward = experience.reward
        action = goal_candidates[best_goal_ix]
        done = experience.done

        assert next_state.shape[0] == self.state_size

        return state, action, reward, next_state, done


    def log_probability_lower_level_picks_action(self, state, goal, action):
        """Calculates the log probability that the lower level agent would have chosen this action given the state
        and goal as inputs"""
        state_and_goal = torch.from_numpy(np.concatenate((state, goal))).float().unsqueeze(0).to(self.device)
        action_would_have_taken = self.lower_level_policy(state_and_goal).detach()
        return -0.5 * torch.norm(action - action_would_have_taken, 2)**2


class Higher_Level_Agent_Environment_Wrapper(Wrapper):
    """Adapts the game environment so that it is compatible with the higher level agent which sets goals for the lower
    level agent"""
    def __init__(self, env, HIRO_agent):
        Wrapper.__init__(self, env)
        self.env = env
        self.HIRO_agent = HIRO_agent
        self.action_space = self.observation_space


    def reset(self, **kwargs):
        self.HIRO_agent.higher_level_state = self.env.reset(**kwargs)
        return self.HIRO_agent.higher_level_state

    def step(self, goal):
        self.HIRO_agent.higher_level_reward = 0
        self.HIRO_agent.step_lower_level_states = []
        self.HIRO_agent.step_lower_level_action_seen = []

        self.HIRO_agent.goal = goal
        self.HIRO_agent.lower_level_agent.episode_number = 0 #must reset lower level agent to 0 episodes completed otherwise won't run more episodes
        self.HIRO_agent.lower_level_agent.run_n_episodes(num_episodes=1, show_whether_achieved_goal=False, save_and_print_results=False)

        self.HIRO_agent.save_higher_level_experience()

        return self.HIRO_agent.higher_level_next_state, self.HIRO_agent.higher_level_reward, self.HIRO_agent.higher_level_done, {}

class Lower_Level_Agent_Environment_Wrapper(Wrapper):
    """Open AI gym wrapper to help create an environment where a goal from a higher-level agent is treated as part
    of the environment state"""
    def __init__(self, env, HIRO_agent, max_sub_policy_timesteps):
        Wrapper.__init__(self, env)
        self.env = env
        self.meta_agent = HIRO_agent
        self.max_sub_policy_timesteps = max_sub_policy_timesteps

        self.track_intrinsic_rewards = []

    def reset(self, **kwargs):
        if self.meta_agent.higher_level_state is not None: state = self.meta_agent.higher_level_state
        else:
            print("INITIATION ONLY")
            state = self.env.reset()

        if self.meta_agent.goal is not None: goal = self.meta_agent.goal
        else:
            print("INITIATION ONLY")
            goal = state

        self.lower_level_timesteps = 0
        self.meta_agent.lower_level_done = False

        self.meta_agent.lower_level_state = self.turn_internal_state_to_external_state(state, goal)

        return self.meta_agent.lower_level_state

    def turn_internal_state_to_external_state(self, internal_state, goal):
        return np.concatenate((np.array(internal_state), goal))

    def step(self, action):
        import random
        if random.random() < 0.008:
            print("Rolling intrinsic rewards {}".format(np.mean(self.track_intrinsic_rewards[-100:])))


        self.meta_agent.step_lower_level_states.append(self.meta_agent.lower_level_state)
        self.meta_agent.step_lower_level_action_seen.append(action)

        self.lower_level_timesteps += 1
        next_state, extrinsic_reward, done, _ = self.env.step(action)



        self.update_rewards(extrinsic_reward, next_state)
        self.update_goal(next_state)
        self.update_state_and_next_state(next_state)
        self.update_done(done)

        return self.meta_agent.lower_level_next_state, self.meta_agent.lower_level_reward, self.meta_agent.lower_level_done, _

    def update_rewards(self, extrinsic_reward, next_state):
        self.meta_agent.higher_level_reward += extrinsic_reward
        self.meta_agent.lower_level_reward = self.calculate_intrinsic_reward(self.meta_agent.higher_level_state,
                                                                             next_state,
                                                                             self.meta_agent.goal)
    def update_goal(self, next_state):

        self.meta_agent.goal = HIRO.goal_transition(self.meta_agent.higher_level_state, self.meta_agent.goal,
                                                               next_state)

    def update_state_and_next_state(self, next_state):
        self.meta_agent.higher_level_next_state = next_state
        self.meta_agent.lower_level_next_state = self.turn_internal_state_to_external_state(next_state,
                                                                                            self.meta_agent.goal)
        self.meta_agent.higher_level_state = self.meta_agent.higher_level_next_state
        self.meta_agent.lower_level_state = self.meta_agent.lower_level_next_state

    def update_done(self, done):
        self.meta_agent.higher_level_done = done
        self.meta_agent.lower_level_done = done or self.lower_level_timesteps >= self.max_sub_policy_timesteps


    def calculate_intrinsic_reward(self, internal_state, internal_next_state, goal):
        """Calculates the intrinsic reward for the agent according to whether it has made progress towards the goal
        or not since the last timestep"""
        desired_next_state = internal_state + goal
        error = desired_next_state - internal_next_state
        intrinsic_reward = -(np.dot(error, error))**0.5

        self.track_intrinsic_rewards.append(intrinsic_reward)

        return intrinsic_reward






