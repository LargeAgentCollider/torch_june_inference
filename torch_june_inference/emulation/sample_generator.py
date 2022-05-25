import torch
import yaml
import pickle
from tqdm import tqdm
from pyDOE import lhs

from torch_june import Runner
from torch_june.mpi_setup import mpi_rank, mpi_size, mpi_comm


class SampleGenerator:
    def __init__(self, runner, n_samples, parameters_to_vary, save_path):
        self.device = runner.device
        self.parameters_to_vary = parameters_to_vary
        self.runner = runner
        self.save_path = save_path
        self.n_samples = n_samples

    @classmethod
    def from_parameters(cls, params):
        runner = Runner.from_file(params["june_configuration_file"])
        return cls(
            runner=runner,
            n_samples=params["n_samples"],
            parameters_to_vary=params["parameters_to_vary"],
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
        samples_x = torch.tensor(
            lhs(n_dims, samples=self.n_samples, criterion="center"), device=self.device
        )
        for i, (key, param_range) in enumerate(self.parameters_to_vary.items()):
            samples_x[:, i] = (
                samples_x[:, i] * (param_range[1] - param_range[0]) + param_range[0]
            )
        return samples_x

    def run_models(self, samples_x):
        n_samples = len(samples_x)
        low = int(mpi_rank * n_samples / mpi_size)
        high = int((mpi_rank + 1) * n_samples / mpi_size)
        samples_x = samples_x[low:high, :]
        samples_y = None
        for i in tqdm(range(len(samples_x))):
            results = self.run_model(samples_x[i, :])
            tosave = results["cases_per_timestep"]
            if samples_y is None:
                samples_y = tosave.reshape(1, -1)
            else:
                samples_y = torch.vstack((samples_y, tosave))
        samples_y = samples_y / self.runner.n_agents
        #samples_x = samples_x.cpu().numpy()
        #samples_y = samples_y.cpu().numpy()
        #mpi_comm.Barrier()

        return samples_x, samples_y

    def run_model(self, sample_x):
        self.runner.reset_model()
        with torch.no_grad():
            state_dict = self.runner.model.state_dict()
            for i, key in enumerate(self.parameters_to_vary):
                value = sample_x[i]
                state_dict[key].copy_(value)
            results = self.runner.run()
        return results

    def run(self):
        samples_x = self.sample_parameters()
        samples_x, samples_y = self.run_models(samples_x)
        with open(self.save_path, "wb") as f:
            pickle.dump({"samples_x": samples_x, "samples_y": samples_y}, f)
