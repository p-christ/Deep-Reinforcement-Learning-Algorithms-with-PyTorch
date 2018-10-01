from Utility_Functions import load_obj
import numpy as np

results = load_obj("Results_Data2.pkl")
print(results["DQN_Agent"][1])

rolling_scores = results["DQN_Agent"][1]
time_taken = results["DQN_Agent"][2]


starting_point = time_taken / len(rolling_scores)
time_axes = [starting_point * (t + 1.0)  for t in range(len(rolling_scores))]

print(time_axes)


print(time_taken)

print(np.arange(time_taken))