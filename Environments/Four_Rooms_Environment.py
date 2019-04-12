import copy
import random
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot
from Environments.Base_Environment import Base_Environment
from random import randint

class Four_Rooms_Environment(Base_Environment):
    """Four rooms game environment as described in paper http://www-anw.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf"""
    environment_name = "Four Rooms"

    def __init__(self, grid_width=13, grid_height=13, stochastic_actions_probability=1.0/3.0):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.blank_space_name = "    "
        self.wall_space_name = "WALL"
        self.user_space_name = "USER"
        self.goal_space_name = "GOAL"
        self.stochastic_actions_probability = stochastic_actions_probability
        self.actions = set(range(4))
        # Note that the indices of the grid are such that (0, 0) is the top left point
        self.action_to_effect_dict = {0: "North", 1: "East", 2: "South", 3:"West"}
        self.current_user_location = None
        self.current_goal_location = None
        self.reward_for_completing_game = float(int((grid_width + grid_height)*10.0*stochastic_actions_probability))
        self.reward_for_every_move_that_doesnt_complete_game = -1.0
        self.reset_environment()

    def reset_environment(self):
        """Resets the environment and returns the start state"""
        self.grid = self.create_grid()
        self.place_agent()
        self.place_goal()
        self.step_count = 0
        self.state = [self.location_to_state(self.current_user_location), self.location_to_state(self.current_goal_location)]
        return np.array(self.state)

    def conduct_action(self, desired_action):
        if type(desired_action) is np.ndarray:
            assert desired_action.shape[0] == 1
            assert len(desired_action.shape) == 1
            desired_action = desired_action[0]

        self.step_count += 1
        action = self.determine_which_action_will_actually_occur(desired_action)
        desired_new_state = self.calculate_desired_new_state(action)
        if not self.is_a_wall(desired_new_state):
            self.move_user(self.current_user_location, desired_new_state)
        self.next_state = self.location_to_state(self.current_user_location)

        if self.user_at_goal_location():
            self.reward = self.reward_for_completing_game
            self.done = True
        else:
            self.reward = self.reward_for_every_move_that_doesnt_complete_game
            if self.step_count >= self.get_max_steps_per_episode(): self.done = True
            else: self.done = False
        self.state = self.next_state

    def determine_which_action_will_actually_occur(self, desired_action):
        """Chooses what action will actually occur. Gives 1. - self.stochastic_actions_probability chance to the
        desired action occuring and the rest of probability spread equally among the other actions"""
        if random.random() < self.stochastic_actions_probability:
            valid_actions = [action for action in self.actions if action != desired_action]
            action = random.choice(valid_actions)
        else: action = desired_action
        return action

    def calculate_desired_new_state(self, action):
        """Calculates the desired new state on basis of action we are going to do"""
        if action == 0:
            desired_new_state = (self.current_user_location[0] - 1, self.current_user_location[1])
        elif action == 1:
            desired_new_state = (self.current_user_location[0], self.current_user_location[1] + 1)
        elif action == 2:
            desired_new_state = (self.current_user_location[0] + 1, self.current_user_location[1])
        elif action == 3:
            desired_new_state = (self.current_user_location[0], self.current_user_location[1] - 1)
        else:
            raise ValueError("Action must be 0, 1, 2, or 3")
        return desired_new_state

    def move_user(self, current_location, new_location):
        """Moves a user from current location to new location"""
        assert self.grid[current_location[0]][current_location[1]] == self.user_space_name
        self.grid[new_location[0]][new_location[1]] = self.user_space_name
        self.grid[current_location[0]][current_location[1]] = self.blank_space_name
        self.current_user_location = (new_location[0], new_location[1])

    def move_goal(self, current_location, new_location):
        """Moves the goal state from current location to new location"""
        assert self.grid[current_location[0]][current_location[1]] == self.goal_space_name
        self.grid[new_location[0]][new_location[1]] = self.goal_space_name
        self.grid[current_location[0]][current_location[1]] = self.blank_space_name
        self.current_goal_location = (new_location[0], new_location[1])

    def is_a_wall(self, location):
        """Returns boolean indicating whether provided location is a wall or not"""
        return self.grid[location[0]][location[1]] == "WALL"

    def return_num_possible_states(self):
        """Returns the number of possible states in this game"""
        return self.grid_width * self.grid_height

    def user_at_goal_location(self):
        """Returns boolean indicating whether user at goal location"""
        return self.current_user_location == self.current_goal_location

    def location_to_state(self, location):
        """Maps a (x, y) location to an integer that uniquely represents its position"""
        return location[0] + location[1] * self.grid_height

    def state_to_location(self, state):
        """Maps a state integer to the (x, y) grid point it represents"""
        col = int(state / self.grid_height)
        row = state - col*self.grid_height
        return (row, col)

    def create_grid(self):
        """Creates and returns the initial gridworld"""
        grid = [[self.blank_space_name for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        centre_col = int(self.grid_width / 2)
        centre_row = int(self.grid_height / 2)
        gaps = [(centre_row, int(centre_col / 2) - 1),  (centre_row, centre_col + int(centre_col / 2)),
                 (int(centre_row/2), centre_col),(centre_row + int(centre_row/2) + 1, centre_col)]
        for row in range(self.grid_height):
            for col in range(self.grid_width):
                if row == 0 or col == 0 or row == self.grid_height - 1 or col == self.grid_width - 1:
                    grid[row][col] = self.wall_space_name
                if row == centre_row or col == centre_col:
                    grid[row][col] = self.wall_space_name
                if (row , col) in gaps:
                    grid[row][col] = self.blank_space_name
        return grid

    def place_agent(self):
        """Places the agent on a random non-wall square"""
        self.current_user_location = self.randomly_place_something(self.user_space_name, [self.wall_space_name])

    def place_goal(self):
        """Places the goal on a random non-WALL and non-USER square"""
        self.current_goal_location = self.randomly_place_something(self.goal_space_name, [self.wall_space_name, self.user_space_name])

    def randomly_place_something(self, thing_name, invalid_places):
        """Randomly places a thing called thing_name on any square that doesn't have an invalid item on it"""
        thing_placed = False
        while not thing_placed:
            random_row = randint(0, self.grid_height - 1)
            random_col = randint(0, self.grid_width - 1)
            if self.grid[random_row][random_col] not in invalid_places:
                self.grid[random_row][random_col] = thing_name
                thing_placed = True
        return (random_row, random_col)

    def print_current_grid(self):
        """Prints out the grid"""
        for row in range(len(self.grid)):
            print(self.grid[row])

    def visualise_current_grid(self):
        """Visualises the current grid"""
        copied_grid = copy.deepcopy(self.grid)
        for row in range(self.grid_height):
            for col in range(self.grid_width):
                if copied_grid[row][col] == self.wall_space_name:
                    copied_grid[row][col] = -100
                elif copied_grid[row][col] == self.blank_space_name:
                    copied_grid[row][col] = 0
                elif copied_grid[row][col] == self.user_space_name:
                    copied_grid[row][col] = 10
                elif copied_grid[row][col] == self.goal_space_name:
                    copied_grid[row][col] = 20
                else:
                    raise ValueError("Invalid values on the grid")
        copied_grid = np.array(copied_grid)
        cmap = mpl.colors.ListedColormap(["black", "white", "blue", "red"])
        bounds = [-101, -1, 1, 11, 21]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        pyplot.imshow(copied_grid, interpolation='nearest',
                            cmap=cmap, norm=norm)
        print("Black = wall, White = empty, Blue = user, Red = goal")
        pyplot.show()

    def get_action_size(self):
        return 4

    def get_action_types(self):
        return "DISCRETE"

    def get_done(self):
        return self.done

    def get_max_steps_per_episode(self):
        return self.reward_for_completing_game

    def get_next_state(self):
        return np.array(self.next_state)

    def get_reward(self):
        return self.reward

    def get_rolling_period_to_calculate_score_over(self):
        return 100

    def get_score_to_win(self):
        return 0.0

    def get_state_size(self):
        return 1

    def get_num_possible_states(self):
        """Returns the number of possible states there are"""
        return self.grid_height * self.grid_width