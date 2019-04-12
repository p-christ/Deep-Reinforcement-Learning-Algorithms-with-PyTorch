import numpy as np
from Utilities.Data_Structures.Config import Config
from Utilities.Trainer import Trainer


def test_get_mean_and_standard_deviation_difference_results():
    """Tests that get_mean_and_standard_deviation_difference_results method produces correct output"""
    results = [ [1.0, 2.0, 3.0], [5.0, -33.0, 55.0], [2.5, 2.5, 2.5]]
    mean_results = [np.mean([1.0, 5.0, 2.5]), np.mean([2.0, -33.0, 2.5]), np.mean([3.0, 55.0, 2.5])]
    std_results = [np.std([1.0, 5.0, 2.5]), np.std([2.0, -33.0, 2.5]), np.std([3.0, 55.0, 2.5])]
    mean_minus_1_std = [ mean - std_val for mean, std_val in zip(mean_results, std_results)]
    mean_plus_1_std = [ mean + std_val for mean, std_val in zip(mean_results, std_results)]
    config = Config()
    config.standard_deviation_results = 1.0
    trainer = Trainer(config, [])
    mean_minus_x_std_guess, mean_results_guess, mean_plus_x_std_guess = trainer.get_mean_and_standard_deviation_difference_results(results)
    assert mean_results == mean_results_guess
    assert mean_minus_1_std == mean_minus_x_std_guess
    assert mean_plus_1_std == mean_plus_x_std_guess

    config.standard_deviation_results = 3.0
    trainer = Trainer(config, [])
    mean_minus_x_std_guess, mean_results_guess, mean_plus_x_std_guess = trainer.get_mean_and_standard_deviation_difference_results(results)
    mean_plus_3_std = [mean + 3.0*std_val for mean, std_val in zip(mean_results, std_results)]
    mean_minus_3_std = [mean - 3.0*std_val for mean, std_val in zip(mean_results, std_results)]
    assert mean_results == mean_results_guess
    assert mean_minus_3_std == mean_minus_x_std_guess
    assert mean_plus_3_std == mean_plus_x_std_guess

def test_add_default_hyperparameters_if_not_overriden():
    """Tests that add_default_hyperparameters_if_not_overriden function works"""
    config = Config()
    default_hyperparameter_set = {'output_activation': 'None', 'hidden_activations': 'relu', 'dropout': 0.0, 'initialiser': 'default',
     'batch_norm': False, 'columns_of_data_to_be_embedded': [], 'embedding_dimensions': [], 'y_range': (),
     }
    alternative_hyperparmater_set = {'output_activation': "YESSS!!", 'hidden_activations': 'relu', 'dropout': 0.0, 'initialiser': 'default',
     'batch_norm': False, 'columns_of_data_to_be_embedded': [], 'embedding_dimensions': [], 'y_range': (),
     "helo": 20}

    config.hyperparameters = {"DQN_Agents": {}}
    config.hyperparameters = Trainer(config, []).add_default_hyperparameters_if_not_overriden(config.hyperparameters)
    assert config.hyperparameters == {"DQN_Agents": default_hyperparameter_set}

    config.hyperparameters = {"DQN_Agents": {}, "Test": {}}
    config.hyperparameters = Trainer(config, []).add_default_hyperparameters_if_not_overriden(config.hyperparameters)
    assert config.hyperparameters == {"DQN_Agents": default_hyperparameter_set, "Test": default_hyperparameter_set}

    config.hyperparameters = {"DQN_Agents": {"helo": 20,  "output_activation": "YESSS!!"}}
    config.hyperparameters = Trainer(config, []).add_default_hyperparameters_if_not_overriden(config.hyperparameters)
    assert config.hyperparameters == {"DQN_Agents": alternative_hyperparmater_set}

    config.hyperparameters = {"A": {"B": {"helo": 20,  "output_activation": "YESSS!!"}, "C": 10, "D": {} } }
    config.hyperparameters = Trainer(config, []).add_default_hyperparameters_if_not_overriden(config.hyperparameters)
    assert config.hyperparameters == {"A": {"C": 10, "D": default_hyperparameter_set, "B": alternative_hyperparmater_set}}




