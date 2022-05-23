import torch
import pyro
import pandas as pd

from torch_june_inference.inference.base import InferenceEngine


class Pyro(InferenceEngine):
    def pyro_model(self, y_obs):
        with torch.no_grad():
            state_dict = self.runner.model.state_dict()
            for i, key in enumerate(self.priors):
                value = pyro.sample(key, self.priors[key]).to(self.device)
                state_dict[key].copy_(value)
        self.runner.run()
        # Compare to data
        n_agents = self.runner.n_agents
        y = self.runner.results[self.data_observable][self.time_stamps] / n_agents
        y_obs = y_obs[self.time_stamps] / n_agents
        pyro.sample(
            self.data_observable,
            self.likelihood(y),
            obs=y_obs,
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
        mcmc_kernel = pyro.infer.NUTS(self.pyro_model)
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
