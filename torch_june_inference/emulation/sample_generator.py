import torch
import yaml
import pickle
import numpy as np
from tqdm import tqdm
from pyDOE import lhs

from torch_june import Runner
from torch_june.utils import fix_seed
from torch_june_inference.mpi_setup import mpi_rank, mpi_size, mpi_comm, MPI

fix_seed(0)


class SampleGenerator:
    def __init__(
        self,
        runner,
        n_samples,
        parameters_to_vary,
        n_samples_per_parameter,
        save_path,
    ):
        self.device = runner.device
        self.parameters_to_vary = parameters_to_vary
        self.runner = runner
        self.save_path = save_path
        self.n_samples = n_samples
        self.n_samples_per_parameter = n_samples_per_parameter

    @classmethod
    def from_parameters(cls, params):
        runner = Runner.from_file(params["june_configuration_file"])
        return cls(
            runner=runner,
            n_samples=params["n_samples"],
            parameters_to_vary=params["parameters_to_vary"],
            n_samples_per_parameter=params["n_samples_per_parameter"],
            save_path=params["save_path"],
        )

    @classmethod
    def from_file(cls, fpath):
        with open(fpath, "r") as f:
            params = yaml.safe_load(f)
        # reads mpi setup
        return cls.from_parameters(params)

    def sample_parameters(self):
        n_dims = len(self.parameters_to_vary)
        parameters = torch.tensor(
            lhs(n_dims, samples=self.n_samples, criterion="center"), device=self.device
        )
        for i, (key, param_range) in enumerate(self.parameters_to_vary.items()):
            parameters[:, i] = (
                parameters[:, i] * (param_range[1] - param_range[0]) + param_range[0]
            )
        return parameters

    def run_models(self, parameters):
        n_samples = len(parameters)
        low = int(mpi_rank * n_samples / mpi_size)
        high = int((mpi_rank + 1) * n_samples / mpi_size)
        parameters = parameters[low:high, :]
        means = None
        stds = None
        for i in tqdm(range(len(parameters))):
            results_array = None
            for j in range(self.n_samples_per_parameter):
                results = self.run_model(parameters[i, :])["cases_per_timestep"]
                if results_array is None:
                    results_array = results.reshape(1, -1)
                else:
                    results_array = torch.vstack((results_array, results))
            if means is None:
                means = torch.mean(results_array, 0)
            else:
                means = torch.vstack((means, torch.mean(results_array, 0)))
            if stds is None:
                stds = torch.std(results_array, 0)
            else:
                stds = torch.vstack((stds, torch.std(results_array, 0)))
        return parameters, means, stds

    def run_model(self, sample_x):
        self.runner.reset_model()
        with torch.no_grad():
            state_dict = self.runner.model.state_dict()
            for i, key in enumerate(self.parameters_to_vary):
                value = sample_x[i]
                state_dict[key].copy_(value)
            results = self.runner()
        return results

    def run(self):
        parameters = self.sample_parameters()
        parameters, means, stds = self.run_models(parameters)
        parameters = parameters.cpu().numpy()
        means = means.cpu().numpy()
        stds = stds.cpu().numpy()
        mpi_comm.Barrier()
        for i in range(1, mpi_size):
            if mpi_rank != i:
                continue
            mpi_comm.send(
                {"parameters": parameters, "means": means, "stds": stds}, dest=0, tag=i
            )
        if mpi_rank == 0:
            for i in range(1, mpi_size):
                data = mpi_comm.recv(source=i, tag=i)
                parameters = np.concatenate((parameters, data["parameters"]))
                means = np.concatenate((means, data["means"]))
                stds = np.concatenate((stds, data["stds"]))
            assert len(parameters) == self.n_samples
            self.save_samples(parameters, means, stds)

    def save_samples(self, parameters, means, stds):
        with open(self.save_path, "wb") as f:
            pickle.dump({"parameters": parameters, "means": means, "stds": stds}, f)
