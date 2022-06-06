from pathlib import Path
import torch
import os
import numpy as np
import random
import yaml
from pytest import fixture

from torch_june import Runner


@fixture(autouse=True)
def set_random_seed(seed=999):
    """
    Sets global seeds for testing in numpy, random, and numbaized numpy.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return


@fixture(name="test_path", scope="session")
def get_test_path():
    return Path(os.path.abspath(__file__)).parent


@fixture(name="test_configs_path", scope="session")
def get_test_configs_path(test_path):
    return test_path / "configs"


@fixture(name="test_data_path", scope="session")
def get_test_data_path(test_path):
    return test_path / "data/data.pkl"


@fixture(name="test_results_path", scope="session")
def get_test_results_path(test_path):
    fpath = test_path / "results"
    fpath.mkdir(exist_ok=True, parents=True)
    return fpath

@fixture(name="june_config_path", scope="session")
def get_june_config_path(test_configs_path):
    return test_configs_path / "june.yaml"


@fixture(name="june_config", scope="session")
def get_june_config(june_config_path, test_data_path, test_results_path):
    config = yaml.safe_load(open(june_config_path))
    config["data_path"] = test_data_path
    config["test_save_path"] = test_results_path / "june"
    return config


@fixture(name="runner")
def get_runner(june_config, scope="session"):
    runner = Runner.from_parameters(june_config)
    return runner


@fixture(name="runner_results")
def get_runner_results(runner, scope="session"):
    results = runner()
    runner.save_results(results)
    return results


@fixture(name="june_results_path")
def get_runner_results_path(runner, scope="session"):
    results = runner()
    runner.save_results(results)
    return runner.save_path / "results.csv"
