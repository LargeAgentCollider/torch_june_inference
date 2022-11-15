from abc import ABC
import pandas as pd
import yaml
import pickle
import torch
import gpytorch
from pathlib import Path
import pyro.distributions as dist

from torch_june import Runner
from torch_june_inference.utils import get_attribute, set_attribute
from torch_june_inference.utils import read_device


class InferenceEngine(ABC):
    def __init__(
        self,
        runner,
        priors,
        observed_data,
        data_observable,
        inference_configuration,
        results_path,
        emulator,
        device,
    ):
        super().__init__()
        self.runner = runner
        self.priors = priors
        self.observed_data = observed_data
        self.data_observable = data_observable
        self.inference_configuration = inference_configuration
        self.results_path = self._read_path(results_path)
        self.device = device
        self.emulator = emulator
        if self.emulator:
            self.emulator.set_eval()

    @classmethod
    def from_file(cls, fpath):
        with open(fpath, "r") as f:
            params = yaml.safe_load(f)
        # reads mpi setup
        params["device"] = read_device(params["device"])
        return cls.from_parameters(params)

    @classmethod
    def from_parameters(cls, parameters):
        device = parameters["device"]
        with open(parameters["june_configuration_file"], "r") as f:
            june_params = yaml.safe_load(f)
        june_params["system"]["device"] = parameters["device"]
        runner = Runner.from_parameters(june_params)
        priors = cls.read_parameters_to_fit(parameters)
        observed_data = cls.load_observed_data(parameters)
        data_observable = parameters["data"]["observable"]
        emulator_params = parameters["emulator"]
        if emulator_params.get("use_emulator", False):
            emulator = pickle.load(open(emulator_params["emulator_path"], "rb"))
            emulator = emulator.to(device)
        else:
            emulator = None
        inference_configuration = parameters.get("inference_configuration", {})
        return cls(
            runner=runner,
            priors=priors,
            results_path=parameters["results_path"],
            observed_data=observed_data,
            data_observable=data_observable,
            device=parameters["device"],
            inference_configuration=inference_configuration,
            emulator=emulator,
        )

    @classmethod
    def read_parameters_to_fit(cls, params):
        parameters_to_fit = params["parameters_to_fit"]
        ret = {}
        for key in parameters_to_fit:
            dist_info = parameters_to_fit[key]["prior"]
            dist_class = getattr(dist, dist_info.pop("dist"))
            ret[key] = dist_class(**dist_info)
        return ret

    @classmethod
    def load_observed_data(cls, params):
        data_params = params["data"]
        df = pd.read_csv(data_params["observed_data"], index_col=0)
        ret = {}
        for key in df:
            ret[key] = torch.tensor(df[key], device=params["device"], dtype=torch.float)
        return ret

    def _set_initial_parameters(self):
        with torch.no_grad():
            names_to_save = []
            for param_name in self.priors:
                set_attribute(
                    self.runner.model,
                    param_name,
                    torch.nn.Parameter(self.priors[param_name].loc),
                )
                names_to_save.append(param_name)
        return names_to_save

    def evaluate_emulator(self, samples):
        with gpytorch.settings.fast_pred_var():
            x = (
                torch.cat([samples[key].reshape(1) for key in samples])
                .to(torch.float)
                .reshape(1, -1)
            )
            pred = self.emulator(x)
            mean = pred["means"].flatten()
            std = pred["stds"].flatten()
        # error_emulator = (abs(res - lower) + abs(res - upper)) / 2
        # error = error_emulator.flatten()
        return mean, std

    def evaluate_model(self, samples):
        runner = Runner.from_parameters(self.runner.parameters)
        with torch.no_grad():
            for param_name in samples:
                set_attribute(runner.model, param_name, samples[param_name])
        results = runner()
        return (
            results,
            0.0,
        )

    def evaluate(self, samples):
        if self.emulator:
            res, error = self.evaluate_emulator(samples)
        else:
            res, error = self.evaluate_model(samples)
        return res, error

    def _read_path(self, results_path):
        results_path = Path(results_path)
        results_path.mkdir(exist_ok=True, parents=True)
        return results_path

    def save_results(self, path):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError
