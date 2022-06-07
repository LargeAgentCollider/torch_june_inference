# Markov Chain Monte Carlo

In this section we describe how to configure and perform MCMC inference with TorchJune.

We use the [Pyro](http://pyro.ai/) programming language to automate the complicated stuff for us.

## 1. NUTS

An example configuration file can be found in `configs/pyro.yaml`:

```yaml
title: Pyro example configuration file.

device: "cuda:0"
june_configuration_file: "./configs/june_config.yaml"

results_path: "./pyro_tests"

inference_configuration:
  kernel: 
    type: NUTS
    max_tree_depth: 10
  num_samples: 5000
  warmup_steps: 500
  likelihood: Normal

parameters_to_fit:
  infection_networks.networks.household.log_beta:
    prior:
      dist: Normal
      loc: 0.3
      scale: 0.1
  infection_networks.networks.company.log_beta:
    prior:
      dist: Normal
      loc: 0.1
      scale: 0.1
  infection_networks.networks.school.log_beta:
    prior:
      dist: Normal
      loc: 0.1
      scale: 0.1

emulator:
  use_emulator: true
    #emulator_config_path: "./configs/emulator.yaml"
  emulator_path: "./data/emulator.pkl"

data:
  observable: 
    cases_per_timestep:
      time_stamps: [-1]
      error: 0.002
    
  observed_data: "./june_example/results.csv"
```

Most of the fields are recognisable from previous sections and they have the same meaning here. The `inference_configuration` section takes care of the specificity of the MCMC inference engine. In this case we use the `NUTS` kernel, but we could also use `HMC`, the `max_tree_depth` is set to 10. Lower values will lead to much faster inference but it may produce biased results. The `warmup_steps` denotes the number of burn-in samples to draw, and the `num_samples` parameter is the number of samples of the MCMC chain. We can also specify the type of likelihood function to use, for now only the `Normal` distribution is supported.

The `parameters_to_fit` section denotes the parameters we want to fit. Note that if we are using an emulator, it is important that we have as many parameters here as the emulator takes as input, in the same order they were sampled. We can specify the prior for each of them, any [distribution supported by `Pyro`](https://docs.pyro.ai/en/stable/distributions.html) is allowed. 

We can either do inference using the `TorchJune` model or a trained emulator. This is configures in the `emulator` section, where we need to specify the emulator path if we are using one.

Finally, the `data` section configures the data we are fitting the model to. We can have multiple observables, at different `time_stamps` and with different errors. The errors are considered to be Gaussian. Finally the `observed_data` field is a path to the data file containing a Dataframe where the columns are the observables.

We can then run Pyro by doing

```bash
python scripts/run_pyro.py configs/pyro.yaml
```

The MCMC chain can be easily plotted doing

```python
import matplotlib.pyplot as plt
import pandas as pd

fig, ax = plt.subplots(1, 2, figsize=(10, 4) )

dfw = pd.read_csv("./pyro_tests/pyro_chain_Warmup.csv")
dfs = pd.read_csv("./pyro_tests/pyro_chain_Sample.csv")

dfw.plot(ax=ax[0], title="Warmup", legend=False)
dfs.plot(ax=ax[1], title="Samples", legend=False)
ax[0].legend(loc="center left", bbox_to_anchor=(2.2,0.5))
plt.show()
```

![](/home/arnau/code/torch_june_inference/docs/images/emulation/pyro_chain.png)

Finally, we can use [corner.py](https://corner.readthedocs.io/en/latest/index.html) to plot our posterior estimates

```python
import corner
f = corner.corner(dfs.values, labels = labels, smooth=2, truths=true_values, bins=25, show_titles=True)
```

where we see that our estimates agree with the values in the original `june_config.yaml` file :).

![pyro_chain](/home/arnau/code/torch_june_inference/docs/images/emulation/pyro_posteriors.png)