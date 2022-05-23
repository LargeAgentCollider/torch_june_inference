import torch
import pandas as pd
import yaml
import pickle
from tqdm import tqdm
import numpy as np
import gpytorch

from torch_june.utils import read_device


class GPEmulator(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x,
        train_y,
        likelihood=None,
        device="cpu",
        save_path="./emulator.path",
    ):
        n_tasks = train_y.shape[-1]
        if likelihood is None:
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=n_tasks
            )
        super(GPEmulator, self).__init__(train_x, train_y, likelihood)
        self.likelihood = likelihood.to(device)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(), num_tasks=n_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=n_tasks, rank=1
        )
        self.train_x = train_x
        self.train_y = train_y
        self.save_path = save_path

    @classmethod
    def from_parameters(cls, params):
        train_x, train_y = cls.load_samples(
            params["samples_path"], time_stamps=params["time_stamps"]
        )
        device = params["device"]
        return cls(
            train_x=train_x,
            train_y=train_y,
            device=device,
            save_path=params["save_path"],
        )

    @classmethod
    def from_file(cls, fpath):
        with open(fpath, "r") as f:
            params = yaml.safe_load(f)
        # reads mpi setup
        params["device"] = read_device(params["device"])
        return cls.from_parameters(params)

    @classmethod
    def load_samples(cls, fpath, time_stamps):
        with open(fpath, "rb") as f:
            samples = pickle.load(f)
        train_x = samples["samples_x"].float()
        train_y = samples["samples_y"][:, time_stamps].float()
        return train_x, train_y

    def restore_state(self, fpath):
        state_dict = torch.load(fpath)
        self.load_state_dict(state_dict)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

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
            pbar.set_description(f"{i}: {loss:.3e}\t {previous_loss:.3e}")

            optimizer.step()
            if i % 100 == 0:
                if loss < previous_loss:
                    previous_loss = loss
                else:
                    break

    def set_eval(self):
        self.eval()
        self.likelihood.eval()

    def save(self):
        torch.save(self.state_dict(), self.save_path)
