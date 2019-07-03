import pickle

from utilities.data_structures.Config import Config
from Trainer import Trainer


trainer = Trainer(config=Config(), agents=None)

#
# trainer.visualise_set_of_preexisting_results(save_image_path="Four_Rooms_and_Long_Corridor.png", results_data_paths=["Long_Corridor_Results_Data.pkl", "Four_Rooms.pkl"],
#                                       plot_titles=["Long Corridor", "Four Rooms"], y_limits=[(0.0, 0.25), (-90.0, 100.25)])



trainer.visualise_preexisting_results(save_image_path="hrl_experiments/Taxi_graph_comparison.png", data_path="hrl_experiments/Taxi_data.pkl",
                                      title="Taxi v2", y_limits=(-800.0, 0.0))


# trainer.visualise_preexisting_results(save_image_path="Long_Corridor_Graph.png", data_path="Long_Corridor_Results_Data.pkl",
#                                       title="Long Corridor", y_limits=(0.0, 0.25))


# trainer.visualise_preexisting_results(save_image_path="Hopper_Results_Graph_Both_Agents.png", data_path="Hopper_Results_Data.pkl",
#                                       title="Hopper") #, y_limits=(0.0, 0.25))

# trainer.visualise_set_of_preexisting_results(results_data_paths=["Cart_Pole_Results_Data.pkl",
#                                                                  "Mountain_Car_Results_Data.pkl"],
#                                              plot_titles=["Cart Pole (Discrete Actions)", "Mountain Car (Continuous Actions)"],
#                                              save_image_path="CartPole_and_MountainCar_Graph.png")



# trainer.visualise_set_of_preexisting_results(results_data_paths=["Data_and_Graphs/Bit_Flipping_Results_Data.pkl",
#                                                                  "Data_and_Graphs/Fetch_Reach_Results_Data.pkl"],
#                                              plot_titles=["Bit Flipping", "Fetch Reach"],
#                                              save_image_path="Data_and_Graphs/HER_Experiments.png")
