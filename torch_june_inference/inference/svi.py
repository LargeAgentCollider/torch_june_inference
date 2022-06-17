import torch
import pandas as pd
import pyro
from tqdm import tqdm

from torch_june_inference.inference.base import InferenceEngine


def get_attribute(base, path):
    paths = path.split(".")
    for p in paths:
        base = getattr(base, p)
    return base


def set_attribute(base, path, target):
    paths = path.split(".")
    _base = base
    for p in paths[:-1]:
        _base = getattr(_base, p)
    setattr(_base, paths[-1], target)


class SVI(InferenceEngine):
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
        super().__init__(
            runner=runner,
            priors=priors,
            observed_data=observed_data,
            data_observable=data_observable,
            inference_configuration=inference_configuration,
            results_path=results_path,
            emulator=emulator,
            device=device,
        )
        pyro.nn.module.to_pyro_module_(self.runner)

    def _get_optimizer(self):
        config = self.inference_configuration["optimizer"]
        optimizer_class = getattr(pyro.optim, config.pop("type"))
        optimizer = optimizer_class(config)
        return optimizer

    def _get_loss(self):
        config = self.inference_configuration["loss"]
        loss_class = getattr(pyro.infer, config.pop("type"))
        loss = loss_class(**config)
        return loss

    def model(self, y_obs):
        # samples = {}
        for prior, value in self.priors.items():
            beta_name = prior.split(".")[-2]
            beta = pyro.sample(f"beta_{beta_name}", value)
            # samples[prior] = beta
            # set_attribute(self.runner.model, prior, pyro.nn.module.PyroSample(value))
            set_attribute(self.runner.model, prior, beta)
        # y, model_error = self.evaluate(samples)
        y = self.runner()
        for key in self.data_observable:
            time_stamps = self.data_observable[key]["time_stamps"]
            data = y[key][time_stamps]
            data_obs = y_obs[key][time_stamps]
            # print(f"data {data}")
            # print(f"data_obs {data_obs}")
            rel_error = self.data_observable[key]["error"]
            # print(f"error {rel_error * data}")
            # print("----")
            for i in pyro.plate("plate_obs", len(time_stamps)):
                pyro.sample(
                    f"obs_{i}",
                    pyro.distributions.Normal(data[i], 0.15 * data[i]),
                    # pyro.distributions.Delta(data[i]),
                    obs=data_obs[i],
                )

    def guide(self, data):
        # samples = {}
        for prior, value in self.priors.items():
            beta_name = prior.split(".")[-2]
            beta_mu = torch.randn_like(get_attribute(self.runner.model, prior))
            beta_mu_param = pyro.param(f"beta_mu_{beta_name}", beta_mu)
            beta_sigma = torch.randn_like(get_attribute(self.runner.model, prior))
            beta_sigma_param = torch.nn.functional.softplus(
                pyro.param(f"beta_sigma_{beta_name}", beta_sigma)
            )
            beta_prior = pyro.distributions.Normal(
                loc=beta_mu_param, scale=beta_sigma_param
            )
            beta = pyro.sample(f"beta_{beta_name}", beta_prior)
            # samples[prior] = beta
            set_attribute(self.runner.model, prior, beta)
        y = self.runner()
        # y, model_error = self.evaluate(samples)
        return y

    def model_emulator(self, y_obs):
        samples = {}
        for prior, value in self.priors.items():
            beta_name = prior.split(".")[-2]
            beta = pyro.sample(f"beta_{beta_name}", value).to(self.device)
            samples[prior] = beta
        y, model_error = self.evaluate_emulator(samples)
        for key in self.data_observable:
            time_stamps = self.data_observable[key]["time_stamps"]
            data = y
            data_obs = y_obs[key][time_stamps]
            rel_error = self.data_observable[key]["error"]
            for i in pyro.plate("plate_obs", len(time_stamps)):
                pyro.sample(
                    f"obs_{i}",
                    pyro.distributions.Normal(data[i], model_error),
                    obs=data_obs[i],
                )

    def guide_emulator(self, data):
        samples = {}
        for prior, value in self.priors.items():
            beta_name = prior.split(".")[-2]
            beta_mu = torch.randn_like(get_attribute(self.runner.model, prior))
            beta_mu_param = pyro.param(f"beta_mu_{beta_name}", beta_mu)
            beta_sigma = torch.randn_like(get_attribute(self.runner.model, prior))
            beta_sigma_param = torch.nn.functional.softplus(
                pyro.param(f"beta_sigma_{beta_name}", beta_sigma)
            )
            beta_prior = pyro.distributions.Normal(
                loc=beta_mu_param, scale=beta_sigma_param
            )
            beta = pyro.sample(f"beta_{beta_name}", beta_prior).to(self.device)
            samples[prior] = beta
        y, model_error = self.evaluate_emulator(samples)
        return y

    def _init_df(self):
        columns = ["loss"]
        for prior in self.priors:
            beta_name = prior.split(".")[-2]
            columns += [f"beta_mu_{beta_name}", f"beta_sigma_{beta_name}"]
        df = pd.DataFrame(columns=columns)
        return df

    def run(self):
        optimizer = self._get_optimizer()
        loss = self._get_loss()
        df = self._init_df()
        data = self.observed_data
        svi = pyro.infer.SVI(
            self.model_emulator, self.guide_emulator, optimizer, loss=loss
        )
        n_steps = self.inference_configuration["n_steps"]
        param_store = pyro.get_param_store()
        for step in tqdm(range(n_steps)):
            loss = svi.evaluate_loss(data)
            svi.step(data)
            df.loc[step, "loss"] = loss
            for param in param_store:
                if param not in df.columns:
                    continue
                df.loc[step, param] = param_store[param].item()
            if step % 10 == 0:
                df.to_csv(self.results_path / f"svi_results.csv")
