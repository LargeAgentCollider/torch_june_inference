import torch
import pandas as pd
import yaml
import pickle
from tqdm import tqdm
import numpy as np
import gpytorch

from torch_june_inference.utils import read_device


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, device, likelihood=None):
        if likelihood is None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.likelihood = likelihood
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.train_x = train_x
        self.train_y = train_y
        self.to(device)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def restore_state(self, fpath):
        state_dict = torch.load(fpath)
        self.load_state_dict(state_dict)

    def train_emulator(self, optimizer=None, max_training_iter=100):
        self.train()
        self.likelihood.train()
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        previous_loss = np.inf
        pbar = tqdm(range(max_training_iter))
        for i in pbar:
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self(self.train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, self.train_y)
            if i >= max_training_iter:
                break
            loss.backward()
            pbar.set_description(f"{i}: {loss:.3e} {previous_loss:.2e}")

            optimizer.step()
            if i % 100 == 0:
                if loss < previous_loss:
                    previous_loss = loss
                else:
                    break

    def set_eval(self):
        self.eval()
        self.likelihood.eval()


class GPEmulator(torch.nn.Module):
    def __init__(
        self,
        parameters,
        means,
        stds,
        device="cpu",
        save_path="./emulator.pkl",
    ):
        super().__init__()
        self.parameters = parameters
        self.means = means
        self.stds = stds
        self.save_path = save_path
        self.device = device
        self.emulators = self._init_emulators()

    @classmethod
    def from_parameters(cls, params):
        device = params["device"]
        parameters, means, stds = cls.load_samples(
            params["emulator_configuration"]["samples_path"],
            time_stamps=params["emulator_configuration"]["time_stamps"],
            device=device,
        )
        return cls(
            parameters=parameters,
            means=means,
            stds=stds,
            device=device,
            save_path=params["emulator_configuration"]["save_path"],
        )

    @classmethod
    def from_file(cls, fpath):
        with open(fpath, "r") as f:
            params = yaml.safe_load(f)
        # reads mpi setup
        params["device"] = read_device(params["device"])
        return cls.from_parameters(params)

    @classmethod
    def load_samples(cls, fpath, time_stamps, device):
        with open(fpath, "rb") as f:
            samples = pickle.load(f)
        parameters = samples["parameters"].float().to(device)
        means = samples["means"][:, time_stamps].float().to(device)
        stds = samples["stds"][:, time_stamps].float().to(device)
        return parameters, means, stds

    def _init_emulators(self):
        ret = {"means": [], "stds": []}
        assert self.means.shape == self.stds.shape
        for i in range(self.means.shape[-1]):
            mean_emulator = ExactGPModel(
                self.parameters, self.means[:, i], device=self.device
            )
            ret["means"].append(mean_emulator)
            std_emulator = ExactGPModel(
                self.parameters, self.stds[:, i], device=self.device
            )
            ret["stds"].append(std_emulator)
        return ret

    def forward(self, x):
        ret = {
            "means": torch.zeros((0, 1), device=self.device),
            "stds": torch.zeros((0, 1), device=self.device),
        }
        for key in self.emulators:
            for emulator in self.emulators[key]:
                pred = emulator.likelihood(emulator(x))[0].loc.flatten()
                ret[key] = torch.vstack((ret[key], pred))
                # ret[key].append(emulator(x))
        return ret

    def train_emulators(self, optimizer=None, max_training_iter=100):
        for key in self.emulators:
            for emulator in self.emulators[key]:
                emulator.train_emulator(
                    optimizer=optimizer, max_training_iter=max_training_iter
                )

    def set_eval(self):
        for key in self.emulators:
            for emulator in self.emulators[key]:
                emulator.set_eval()

    def save(self):
        with open(self.save_path, "wb") as f:
            pickle.dump(self, f)

    # def to(self, device):
    #    for key in self.emulators:
    #        for emulator in self.emulators[key]:
    #            emulator.to(device)
    #            emulator.likelihood.to(device)
    #    return super().to(device)
