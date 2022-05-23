import sys

from torch_june_inference.inference import MultiNest

mn = MultiNest.from_file(sys.argv[1])
mn.run()
