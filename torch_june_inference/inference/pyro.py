import torch
import pyro
import pandas as pd

from torch_june_inference.inference.base import InferenceEngine


class Pyro(InferenceEngine):
    def pyro_model(self, y_obs):
        self.runner.reset_model()
        samples = {}
        for key in self.priors:
            value = pyro.sample(key, self.priors[key]).to(self.device)
            samples[key] = value
        y, model_error = self.evaluate(samples)
        # Compare to data
        y_obs = y_obs[self.time_stamps] / self.runner.n_agents
        for key in self.data_obervable:
            time_stamps = self.data_obervable["time_stamps"]
            data = y[key][time_stamps]
            data_obs = y[key].iloc[time_stamps]
            pyro.sample(
                key,
                self.likelihood(loc=data, scale=model_error),
                obs=data_obs,
            )

    def logger(self, kernel, samples, stage, i, dfs):
        df = dfs[stage]
        for key in samples:
            if "beta" not in key:
                continue
            unconstrained_samples = samples[key].detach()
            constrained_samples = kernel.transforms[key].inv(unconstrained_samples)
            df.loc[i, key] = constrained_samples.cpu().item()
        df.to_csv(self.results_path / f"chain_{stage}.csv", index=False)

    def run(self):
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
