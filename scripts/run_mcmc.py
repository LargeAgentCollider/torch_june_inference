import sys

from torch_june_inference.inference import MCMC

mcmc = MCMC.from_file(sys.argv[1])
mcmc.run()
