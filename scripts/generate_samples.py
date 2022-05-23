from torch_june_inference.emulation import SampleGenerator


sg = SampleGenerator.from_file("./configs/sample_generator.yaml")
sg.run()
