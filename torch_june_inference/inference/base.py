from abc import ABC
import pandas as pd
import yaml
import torch
from pathlib import Path
import pyro.distributions as dist

from torch_june import Runner

def fstr(template):
    return eval(f"f'{template}'")

class InferenceEngine(ABC):
    def __init__(
        self,
        runner,
        priors,
        likelihood,
        observed_data,
        time_stamps,
        data_observable,
        results_path,
        device,
    ):
        super().__init__()
        self.runner = runner
        self.priors = priors
        self.likelihood = likelihood
        self.observed_data = observed_data
        self.time_stamps = time_stamps
        self.data_observable = data_observable
        self.results_path = self._read_path(results_path)
        self.device = fstr(device)

    @classmethod
    def from_file(cls, fpath):
        with open(fpath, "r") as f:
            params = yaml.safe_load(f)
        return cls.from_parameters(params)

    @classmethod
    def from_parameters(cls, parameters):
        runner = Runner.from_file(parameters["june_configuration_file"])
        priors = cls.read_parameters_to_fit(parameters)
        likelihood = cls.initialize_likelihood(parameters)
        observed_data = cls.load_observed_data(parameters)
        time_stamps = parameters["data"]["time_stamps"]
        data_observable = parameters["data"]["observable"]
        return cls(
            runner=runner,
            priors=priors,
            likelihood=likelihood,
            time_stamps=time_stamps,
            results_path=parameters["results_path"],
            observed_data=observed_data,
            data_observable=data_observable,
            device=parameters["device"],
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

    def _read_path(self, results_path):
        results_path = Path(results_path)
        results_path.mkdir(exist_ok=True, parents=True)
        return results_path

    def save_results(self, path):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError
