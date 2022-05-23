import pytest
import numpy as np
import torch
from pathlib import Path

test_path = Path(__file__).parent.parent


from torch_june_inference.inference import MultiNest


class TestMultiNest:
    @pytest.fixture(name="true_parameter")
    def make_true_param(self):
        return torch.tensor([3.0])

    def model(self, param, x):
        return param[0] * torch.sin(x)

    @pytest.fixture(name="data")
    def make_data(self, true_parameter):
        x = torch.linspace(0, 10, 50)
        y = self.model(true_parameter, x) + torch.distributions.Normal(0, 0.3).sample(
            x.shape
        )
        return x, y

    def loglike(self, y, y_obs):
        loglikelihood = (
            torch.distributions.Normal(
                y,
                0.3,
            )
            .log_prob(y_obs)
            .sum()
            .cpu()
            .item()
        )
        return loglikelihood

    @pytest.fixture(name="mn")
    def run_multinest(self, data):
        loglike = lambda cube, ndim, nparams: self.loglike(cube, ndim, nparams, data)
        mn = MultiNest(
            model=self.model,
            prior=self.prior,
            loglike=self.loglike,
            ndim=1,
            output_path=test_path / "multinest_test_results",
        )
        return mn

    def test__multinest(self, mn, data, true_parameter):
        x_obs, y_obs = data
        mn.run(x_obs, y_obs, verbose=False)
        samples = mn.samples
        hist = np.histogram(samples, density=True, bins=100)
        param_estimation = hist[1][np.argmax(hist[0])]
        assert np.isclose(param_estimation, true_parameter.item(), atol=0, rtol=0.05)
