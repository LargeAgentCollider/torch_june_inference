import torch
from tqdm import tqdm
import numpy as np
import gpytorch


class GPEmulator(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood=None, device="cpu"):
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

    def save(self, path):
        torch.save(self.state_dict(), path)
