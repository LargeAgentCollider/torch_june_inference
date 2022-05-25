import sys

from torch_june_inference.emulation import SampleGenerator


sg = SampleGenerator.from_file(sys.argv[1])
sg.run()
