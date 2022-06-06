from torch_june.runner import Runner
import torch
import sys

with torch.no_grad():
    runner = Runner.from_file(sys.argv[1])
    results = runner()

n_agents = runner.data["agent"].id.shape[0]
print(results["cases_per_timestep"] * n_agents)
print(results["cases_per_timestep"])

runner.save_results(results)
