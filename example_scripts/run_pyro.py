from torch_june_inference.inference import Pyro

pyro = Pyro.from_file("./configs/pyro.yaml")
pyro.run()
