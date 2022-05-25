import sys

from torch_june_inference.inference import GradientDescent

pyro = GradientDescent.from_file(sys.argv[1])
pyro.run()
