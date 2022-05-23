from torch_june_inference.inference import MultiNest

mn = MultiNest.from_file("./configs/multinest.yaml")
mn.run()
