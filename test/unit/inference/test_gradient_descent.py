from pytest import fixture
import yaml
import numpy as np
import torch
import pandas as pd

from torch_june_inference.inference import GradientDescent


@fixture(name="gd_config")
def get_gd_config(test_configs_path):
    return test_configs_path / "gradient_descent.yaml"


@fixture(name="gd_params")
def make_gd_params(gd_config, june_results_path, june_config_path, test_results_path):
    config = yaml.safe_load(open(gd_config))
    config["june_configuration_file"] = june_config_path
    config["results_path"] = test_results_path / "gd"
    config["data"]["observed_data"] = june_results_path
    return config


@fixture(name="gd")
def make_gd(gd_params):
    # run results
    return GradientDescent.from_parameters(gd_params)


class TestGradientDescent:
    def test__set_initial_parameters(self, gd):
        names_to_save = gd._set_initial_parameters()
        to_fit = set(
            [
                "infection_networks.networks.household.log_beta",
                "infection_networks.networks.school.log_beta",
            ]
        )
        assert set(names_to_save) == to_fit
        for name, param in gd.runner.model.named_parameters():
            if name in to_fit:
                assert param.requires_grad == True
            else:
                assert param.requires_grad == False

    def test__get_optimizer(self, gd):
        optimizer = gd._get_optimizer()
        assert optimizer.__class__.__name__ == "SGD"
        assert optimizer.defaults["momentum"] == 0.1
        assert optimizer.defaults["lr"] == 0.01

    def test__get_loss(self, gd):
        loss = gd._get_loss()
        assert loss.__class__.__name__ == "MSELoss"
        assert loss.reduction == "mean"

    def test__fit(self, gd):
        assert gd.inference_configuration["n_epochs"] == 100
        gd.run(verbose=False)
        results = pd.read_csv(gd.results_path / "training.csv")
        beta_hist = results["infection_networks.networks.household.log_beta"]
        true_beta = 0.5
        previous_distance = abs(true_beta - beta_hist.iloc[0])
        current_distance = abs(true_beta - beta_hist.iloc[-1])
        assert current_distance < previous_distance

        beta_hist = results["infection_networks.networks.school.log_beta"]
        true_beta = 0.2
        previous_distance = abs(true_beta - beta_hist.iloc[0])
        current_distance = abs(true_beta - beta_hist.iloc[-1])
        assert current_distance < previous_distance
