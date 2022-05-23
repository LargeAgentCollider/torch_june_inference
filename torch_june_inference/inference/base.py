from abc import ABC
import pandas as pd
import yaml
import torch
from pathlib import Path
import pyro.distributions as dist

from torch_june import Runner
from torch_june.utils import read_device
from torch_june_inference.emulation import GPEmulator


class InferenceEngine(ABC):
    def __init__(
        self,
        runner,
        priors,
        likelihood,
        observed_data,
        time_stamps,
        data_observable,
        inference_configuration,
        results_path,
        emulator,
        device,
    ):
        super().__init__()
        self.runner = runner
        self.priors = priors
        self.likelihood = likelihood
        self.observed_data = observed_data
        self.time_stamps = time_stamps
        self.data_observable = data_observable
        self.inference_configuration = inference_configuration
        self.results_path = self._read_path(results_path)
        self.device = device
        self.emulator = emulator

    @classmethod
    def from_file(cls, fpath):
        with open(fpath, "r") as f:
            params = yaml.safe_load(f)
        # reads mpi setup
        params["device"] = read_device(params["device"])
        return cls.from_parameters(params)

    @classmethod
    def from_parameters(cls, parameters):
        with open(parameters["june_configuration_file"], "r") as f:
            june_params = yaml.safe_load(f)
            june_params["system"]["device"] = parameters["device"]
        runner = Runner.from_parameters(june_params)
        priors = cls.read_parameters_to_fit(parameters)
        likelihood = cls.initialize_likelihood(parameters)
        observed_data = cls.load_observed_data(parameters)
        time_stamps = parameters["data"]["time_stamps"]
        data_observable = parameters["data"]["observable"]
        emulator_params = parameters["emulator"]
        if emulator_params.get("use_emulator", False):
            emulator = cls.load_emulator(emulator_params)
        else:
            emulator = None
        inference_configuration = parameters.get("inference_configuration", {})
        return cls(
            runner=runner,
            priors=priors,
            likelihood=likelihood,
            time_stamps=time_stamps,
            results_path=parameters["results_path"],
            observed_data=observed_data,
            data_observable=data_observable,
            device=parameters["device"],
            inference_configuration=inference_configuration,
            emulator=emulator
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
    def initialize_likelihood(cls, params):
        lh_params = params["likelihood"]
        dist_class = getattr(dist, lh_params["distribution"])
        error = lh_params["error"]
        return lambda x: dist_class(x, error)

    @classmethod
    def load_observed_data(cls, params):
        data_params = params["data"]
        data_timestamps = data_params["time_stamps"]
        df = pd.read_csv(data_params["observed_data"])
        data = torch.tensor(df[data_params["observable"]], device=params["device"])
        return data

    @classmethod
    def load_emulator(cls, emulator_params):
        emulator = GPEmulator.from_file(emulator_params["emulator_config_path"])
        emulator.restore_state(emulator_params["emulator_path"])
        return emulator

    def evaluate_emulator(self, samples):
        with torch.no_grad():
            test_x = torch.tensor(
                [samples[key] for key in samples],
                device=device,
            )
            observed_pred = likelihood(model(test_x))
            mean = observed_pred.mean
            lower, upper = observed_pred.confidence_region()
        res = mean.flatten() / self.n_agents
        error_emulator = (abs(res - lower) + abs(res - upper)) / 2
        error = error_emulator.flatten() / n_agents + 0.025
        return res, error

    def evaluate_model(self, samples):
        with torch.no_grad():
            state_dict = self.runner.model.state_dict()
            for key in self.priors:
                value = samples[key]
                state_dict[key].copy_(value)
        results = self.runner.run()
        y = results[self.data_observable][self.time_stamps] / self.runner.n_agents
        return y, 0.0

    def evaluate(self, samples):
        if self.emulator:
            res, error = self.evaluate_model(samples)
        else:
            res, error = self.evaluate_emulator(samples)
        return res, error


    def _read_path(self, results_path):
        results_path = Path(results_path)
        results_path.mkdir(exist_ok=True, parents=True)
        return results_path

    def save_results(self, path):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError
