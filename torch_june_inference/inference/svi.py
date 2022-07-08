import torch
import pandas as pd
import pyro
from pyro.nn import PyroSample
from tqdm import tqdm

from torch_june_inference.inference.base import InferenceEngine
from torch_june_inference.utils import set_attribute, get_attribute


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
        self.svi = None
        pyro.nn.module.to_pyro_module_(self.runner)

    def _get_scheduler(self):
        config = self.inference_configuration["optimizer"]
        optimizer_type = config.pop("type")
        if "milestones" in config:
            milestones = config.pop("milestones")
        else:
            milestones = []
        if "gamma" in config:
            gamma = config.pop("gamma")
        else:
            gamma = 1.0
        optimizer_class = getattr(pyro.optim, optimizer_type)
        return optimizer_class(config)
        #scheduler = pyro.optim.MultiStepLR(
        #    {
        #        "optimizer": optimizer_class,
        #        "optim_args": config,
        #        "milestones": milestones,
        #        "gamma": gamma,
        #    }
        #)
        #return scheduler

    def _get_loss(self):
        config = self.inference_configuration["loss"]
        loss_class = getattr(pyro.infer, config.pop("type"))
        loss = loss_class(**config)
        return loss

    def model(self, y_obs):
        state_dict = self.runner.model.state_dict()
        for prior, value in self.priors.items():
            beta_name = prior.split(".")[-2]
            beta = pyro.sample(f"log_beta_{beta_name}", value)
            #with torch.no_grad():
            #    state_dict[prior].copy_(beta)
        y = self.runner()
        for key in self.data_observable:
            time_stamps = self.data_observable[key]["time_stamps"]
            if time_stamps == "all":
                time_stamps = range(len(y[key]))
            data = y[key][time_stamps]
            data_obs = y_obs[key][time_stamps]
            rel_error = self.data_observable[key]["error"]
            data_sq = torch.pow(data, 2.0)
            error = rel_error * torch.sqrt(torch.cumsum(data_sq, dim=0))
            for i in pyro.plate(f"plate_obs_{key}", len(time_stamps)):
                if data[i] == 0:
                    continue
            #    #error = rel_error * data[i]
                pyro.sample(
                    f"obs_{key}_{i}",
                    pyro.distributions.Normal(data[i], error[i]),
                    #pyro.distributions.Delta(data[i]),
                    obs=data_obs[i],
                )

    def guide(self, data):
        state_dict = self.runner.model.state_dict()
        for prior, value in self.priors.items():
            beta_name = prior.split(".")[-2]
            beta_mu_param = pyro.param(f"beta_mu_{beta_name}", value.loc)
            beta_sigma_param = pyro.param(
                f"beta_sigma_{beta_name}",
                torch.tensor(0.1),
                constraint=pyro.distributions.constraints.softplus_positive,
            )
            beta_prior = pyro.distributions.Normal(
                loc=beta_mu_param, scale=beta_sigma_param
            )
            beta = pyro.sample(f"log_beta_{beta_name}", beta_prior)
            set_attribute(self.runner.model, prior, beta)
        y = self.runner()
        return y

    #def model_emulator(self, y_obs):
    #    samples = {}
    #    for prior, value in self.priors.items():
    #        beta_name = prior.split(".")[-2]
    #        beta = pyro.sample(f"beta_{beta_name}", value).to(self.device)
    #        samples[prior] = beta
    #    y, model_error = self.evaluate_emulator(samples)
    #    for key in self.data_observable:
    #        time_stamps = self.data_observable[key]["time_stamps"]
    #        data = y
    #        data_obs = y_obs[key][time_stamps]
    #        rel_error = self.data_observable[key]["error"]
    #        for i in pyro.plate("plate_obs", len(time_stamps)):
    #            pyro.sample(
    #                f"obs_{i}",
    #                pyro.distributions.Normal(data[i], model_error),
    #                obs=data_obs[i],
    #            )

    #def guide_emulator(self, data):
    #    samples = {}
    #    for prior, value in self.priors.items():
    #        beta_name = prior.split(".")[-2]
    #        beta_mu = torch.randn_like(get_attribute(self.runner.model, prior))
    #        beta_mu_param = pyro.param(f"beta_mu_{beta_name}", beta_mu)
    #        beta_sigma = torch.randn_like(get_attribute(self.runner.model, prior))
    #        # beta_sigma_param = torch.nn.functional.softplus(
    #        #    pyro.param(f"beta_sigma_{beta_name}", beta_sigma)
    #        # )
    #        beta_sigma_param = pyro.param(
    #            f"beta_sigma_{beta_name}",
    #            beta_sigma,
    #            constraint=pyro.distributions.constraints.positive,
    #        )
    #        beta_prior = pyro.distributions.Normal(
    #            loc=beta_mu_param, scale=beta_sigma_param
    #        )
    #        beta = pyro.sample(f"beta_{beta_name}", beta_prior).to(self.device)
    #        samples[prior] = beta
    #    y, model_error = self.evaluate_emulator(samples)
    #    return y

    def _init_df(self):
        columns = ["loss"]
        for prior in self.priors:
            beta_name = prior.split(".")[-2]
            columns += [f"beta_mu_{beta_name}", f"beta_sigma_{beta_name}"]
        df = pd.DataFrame(columns=columns)
        return df

    def run(self):
        scheduler = self._get_scheduler()
        loss = self._get_loss()
        df = self._init_df()
        data = self.observed_data
        normal_guide = pyro.infer.autoguide.AutoNormal(self.model)
        self.svi = pyro.infer.SVI(self.model, self.guide, scheduler, loss=loss)
        n_steps = self.inference_configuration["n_steps"]
        param_store = pyro.get_param_store()
        for step in tqdm(range(n_steps)):
            loss = self.svi.evaluate_loss(data)
            self.svi.step(data)
            #scheduler.step()
            df.loc[step, "loss"] = loss
            for param in param_store:
                if "beta" not in param:
                    continue
                df.loc[step, param] = param_store[param].item()
            if step % 10 == 0:
                df.to_csv(self.results_path / f"svi_results.csv")
