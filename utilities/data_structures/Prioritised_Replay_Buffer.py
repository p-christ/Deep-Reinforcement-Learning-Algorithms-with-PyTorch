import numpy as np
import torch
from utilities.data_structures.Deque import Deque
from utilities.data_structures.Max_Heap import Max_Heap

class Prioritised_Replay_Buffer(Max_Heap, Deque):
    """Data structure that maintains a deque, a heap and an array. The deque keeps track of which experiences are the oldest and so
     tells us which ones to delete once the buffer starts getting full. The heap lets us quickly retrieve the experience
     with the max td_value. And the array lets us do quick random samples with probabilities equal to the proportional td errors.
     We also keep track of the sum of the td values using a simple variable.

    NOTE that this implementation is not optimal in terms of speed. At some point I will make improvements to it.

     """

    def __init__(self, hyperparameters, seed=0):
        Max_Heap.__init__(self, hyperparameters["buffer_size"], dimension_of_value_attribute=5, default_key_to_use=0)
        Deque.__init__(self, hyperparameters["buffer_size"], dimension_of_value_attribute=5)
        np.random.seed(seed)

        self.deques_td_errors = self.initialise_td_errors_array()

        self.heap_index_to_overwrite_next = 1
        self.number_experiences_in_deque = 0
        self.adapted_overall_sum_of_td_errors = 0

        self.alpha = hyperparameters["alpha_prioritised_replay"]
        self.beta = hyperparameters["beta_prioritised_replay"]
        self.incremental_td_error = hyperparameters["incremental_td_error"]
        self.batch_size = hyperparameters["batch_size"]

        self.heap_indexes_to_update_td_error_for = None

        self.indexes_in_node_value_tuple = {
            "state": 0,
            "action": 1,
            "reward": 2,
            "next_state": 3,
            "done": 4
        }

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def initialise_td_errors_array(self):
        """Initialises a deque of Nodes of length self.max_size"""
        return np.zeros(self.max_size)

    def add_experience(self, raw_td_error, state, action, reward, next_state, done):
        """Save an experience in the replay buffer"""
        td_error = (abs(raw_td_error) + self.incremental_td_error) ** self.alpha
        self.update_overall_sum(td_error, self.deque[self.deque_index_to_overwrite_next].key)
        self.update_deque_and_deque_td_errors(td_error, state, action, reward, next_state, done)
        self.update_heap_and_heap_index_to_overwrite()
        self.update_number_experiences_in_deque()
        self.update_deque_index_to_overwrite_next()

    def update_overall_sum(self, new_td_error, old_td_error):
        """Updates the overall sum of td_values present in the buffer"""
        self.adapted_overall_sum_of_td_errors += new_td_error  - old_td_error

    def update_deque_and_deque_td_errors(self, td_error, state, action, reward, next_state, done):
        """Updates the deque by overwriting the oldest experience with the experience provided"""
        self.deques_td_errors[self.deque_index_to_overwrite_next] = td_error
        self.add_element_to_deque(td_error, (state, action, reward, next_state, done))

    def add_element_to_deque(self, new_key, new_value):
        """Adds an element to the deque"""
        self.update_deque_node_key_and_value(self.deque_index_to_overwrite_next, new_key, new_value)

    def update_heap_and_heap_index_to_overwrite(self):
        """Updates the heap by rearranging it given the new experience that was just incorporated into it. If we haven't
        reached max capacity then the new experience is added directly into the heap, otherwise a pointer on the heap has
        changed to reflect the new experience so there's no need to add it in"""
        if not self.reached_max_capacity:
            self.update_heap_element(self.heap_index_to_overwrite_next, self.deque[self.deque_index_to_overwrite_next])
            self.deque[self.deque_index_to_overwrite_next].heap_index = self.heap_index_to_overwrite_next
            self.update_heap_index_to_overwrite_next()

        heap_index_change = self.deque[self.deque_index_to_overwrite_next].heap_index
        self.reorganise_heap(heap_index_change)

    def update_heap_index_to_overwrite_next(self):
        """This updates the heap index to write over next. Once the buffer gets full we stop calling this function because
        the nodes the heap points to start being changed directly rather than the pointers on the heap changing"""
        self.heap_index_to_overwrite_next += 1

    def swap_heap_elements(self, index1, index2):
        """Swaps two position of two heap elements and then updates the heap_index stored in the two nodes. We have to override
        this method from Max_Heap so that it also updates the heap_index variables"""
        self.heap[index1], self.heap[index2] = self.heap[index2], self.heap[index1]
        self.heap[index1].heap_index = index1
        self.heap[index2].heap_index = index2

    def sample(self, rank_based=True):
        """Randomly samples a batch from experiences giving a higher likelihood to experiences with a higher td error. It then
        calculates an importance sampling weight for each sampled experience, you can read about this in the paper:
        https://arxiv.org/pdf/1511.05952.pdf"""
        experiences, deque_sample_indexes = self.pick_experiences_based_on_proportional_td_error()
        states, actions, rewards, next_states, dones = self.separate_out_data_types(experiences)
        self.deque_sample_indexes_to_update_td_error_for = deque_sample_indexes
        importance_sampling_weights = self.calculate_importance_sampling_weights(experiences)
        return (states, actions, rewards, next_states, dones), importance_sampling_weights

    def pick_experiences_based_on_proportional_td_error(self):
        """Randomly picks a batch of experiences with probability equal to their proportional td_errors"""
        probabilities = self.deques_td_errors / self.give_adapted_sum_of_td_errors()
        deque_sample_indexes = np.random.choice(range(len(self.deques_td_errors)), size=self.batch_size, replace=False, p=probabilities)
        experiences = self.deque[deque_sample_indexes]
        return experiences, deque_sample_indexes

    def separate_out_data_types(self, experiences):
        """Separates out experiences into their different parts and makes them tensors ready to be used in a pytorch model"""
        states = torch.from_numpy(np.vstack([e.value[self.indexes_in_node_value_tuple["state"]] for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.value[self.indexes_in_node_value_tuple["action"]] for e in experiences])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.value[self.indexes_in_node_value_tuple["reward"]] for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.value[self.indexes_in_node_value_tuple["next_state"]] for e in experiences])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([int(e.value[self.indexes_in_node_value_tuple["done"]]) for e in experiences])).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def calculate_importance_sampling_weights(self, experiences):
        """Calculates the importance sampling weight of each observation in the sample. The weight is proportional to the td_error of the observation,
        see the paper here for more details: https://arxiv.org/pdf/1511.05952.pdf"""
        td_errors = [experience.key for experience in experiences]
        importance_sampling_weights = [((1.0 / self.number_experiences_in_deque) * (self.give_adapted_sum_of_td_errors() / td_error)) ** self.beta for td_error in td_errors]
        sample_max_importance_weight = max(importance_sampling_weights)
        importance_sampling_weights = [is_weight / sample_max_importance_weight for is_weight in importance_sampling_weights]
        importance_sampling_weights = torch.tensor(importance_sampling_weights).float().to(self.device)
        return importance_sampling_weights

    def update_td_errors(self, td_errors):
        """Updates the td_errors for the provided heap indexes. The indexes should be the observations provided most
        recently by the give_sample method"""
        for raw_td_error, deque_index in zip(td_errors, self.deque_sample_indexes_to_update_td_error_for):
            td_error =  (abs(raw_td_error) + self.incremental_td_error) ** self.alpha
            corresponding_heap_index = self.deque[deque_index].heap_index
            self.update_overall_sum(td_error, self.heap[corresponding_heap_index].key)
            self.heap[corresponding_heap_index].key = td_error
            self.reorganise_heap(corresponding_heap_index)
            self.deques_td_errors[deque_index] = td_error

    def give_max_td_error(self):
        """Returns the maximum td error currently in the heap. Because it is a max heap this is the top element of the heap"""
        return self.give_max_key()

    def give_adapted_sum_of_td_errors(self):
        """Returns the sum of td errors of the experiences currently in the heap"""
        return self.adapted_overall_sum_of_td_errors

    def __len__(self):
        """Tells us how many experiences there are in the replay buffer. This number will never exceed self.max_size"""
        return self.number_experiences_in_deque

