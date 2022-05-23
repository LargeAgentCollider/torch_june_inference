import sys

from torch_june_inference.inference import Pyro

pyro = Pyro.from_file(sys.argv[1])
pyro.run()
