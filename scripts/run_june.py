from torch_june.runner import Runner
import torch
import sys

with torch.no_grad():
    runner = Runner.from_file(sys.argv[1])
    results = runner()

n_agents = runner.data["agent"].id.shape[0]
# cases = results["cases_per_timestep"] * n_agents
# cases = cases + torch.distributions.Normal(0, 0.05 * cases).sample()
# results["cases_per_timestep"] = cases / n_agents

#daily_cases = results["daily_cases_per_timestep"] * n_agents
#
#print(daily_cases)
#print("------------")
#daily_cases = daily_cases + torch.distributions.Normal(0, 0.2 * daily_cases).sample()
#results["daily_cases_per_timestep"] = daily_cases / n_agents
#results["cases_per_timestep"] = torch.cumsum(daily_cases, dim=0) / n_agents
#print(results["daily_cases_per_timestep"])
print(results["cases_per_timestep"])

runner.save_results(results)
