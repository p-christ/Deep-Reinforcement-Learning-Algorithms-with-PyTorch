import os

from Utilities import load_obj, visualise_results_by_agent

FILE_TO_SAVE_DATA_RESULTS = "Results_Data.pkl"
FILE_TO_SAVE_RESULTS_GRAPH = "Results_Graph.png"
target_score = 195

agent_number = 1
if os.path.isfile(FILE_TO_SAVE_DATA_RESULTS):
    results = load_obj(FILE_TO_SAVE_DATA_RESULTS)
else:
    results = {}


visualise_results_by_agent(results, target_score, file_to_save_results_graph=FILE_TO_SAVE_RESULTS_GRAPH)

