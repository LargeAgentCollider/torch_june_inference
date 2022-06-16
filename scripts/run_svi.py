import sys

from torch_june_inference.inference import SVI

svi = SVI.from_file(sys.argv[1])
svi.run()
