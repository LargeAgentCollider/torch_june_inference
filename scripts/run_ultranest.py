import sys

from torch_june_inference.inference import UltraNest

mn = UltraNest.from_file(sys.argv[1])
mn.run()
