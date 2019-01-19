

class Config(object):
    """Object to hold the config requirements for an agent/game"""

    def __init__(self):
        self.seed = None
        self.environment = None
        self.requirements_to_solve_game = None
        self.max_episodes_to_run = None
        self.file_to_save_data_results = None
        self.file_to_save_data_results_graph = None
        self.runs_per_agent = None
        self.visualise_overall_results = None
        self.visualise_individual_results = None
        self.hyperparameters = None
        self.use_GPU = None


