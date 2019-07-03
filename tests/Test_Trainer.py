import numpy as np
from utilities.data_structures.Config import Config
from agents.Trainer import Trainer

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





