import torch
import pyro
import pandas as pd
from pyro.infer import infer_discrete

from torch_june_inference.inference.base import InferenceEngine


class MCMC(InferenceEngine):
    def pyro_model(self, y_obs):
        samples = {}
        for key in self.priors:
            value = pyro.sample(key, self.priors[key]).to(self.device)
            samples[key] = value
        y, model_error = self.evaluate(samples)
        likelihood_fn = getattr(
            pyro.distributions, self.inference_configuration["likelihood"]
        )
        for key in self.data_observable:
            time_stamps = self.data_observable[key]["time_stamps"]
            if time_stamps == "all":
                time_stamps = range(len(y[key]))
            data = y[key][time_stamps]
            data_obs = y_obs[key][time_stamps]
            rel_error = 0.2 #self.data_observable[key]["error"]
            data_sq = torch.pow(data, 2.0)
            error = rel_error * torch.sqrt(torch.cumsum(data_sq, dim=0))
            for i in pyro.plate(f"plate_obs_{key}", len(time_stamps)):
                pyro.sample(
                    f"obs_{key}_{i}",
                    pyro.distributions.Normal(data[i], error[i]),
                    obs=data_obs[i],
                )

    def logger(self, kernel, samples, stage, i, dfs):
        df = dfs[stage]
        for key in samples:
            if "beta" not in key:
                continue
            unconstrained_samples = samples[key].detach()
            constrained_samples = kernel.transforms[key].inv(unconstrained_samples)
            df.loc[i, key] = constrained_samples.cpu().item()
        df.to_csv(self.results_path / f"mcmc_chain_{stage}.csv", index=False)

    def run(self):
        names_to_save = self._set_initial_parameters()
        dfs = {"Sample": pd.DataFrame(), "Warmup": pd.DataFrame()}
        kernel_f = getattr(
            pyro.infer, self.inference_configuration["kernel"].pop("type")
        )
        mcmc_kernel = kernel_f(
            self.pyro_model, **self.inference_configuration["kernel"]
        )
        mcmc = pyro.infer.MCMC(
            mcmc_kernel,
            num_samples=self.inference_configuration["num_samples"],
            warmup_steps=self.inference_configuration["warmup_steps"],
            hook_fn=lambda kernel, samples, stage, i: self.logger(
                kernel, samples, stage, i, dfs
            ),
        )
        mcmc.run(self.observed_data)
        print(mcmc.summary())
        print(mcmc.diagnostics())
