import sys

from torch_june_inference.emulation import GPEmulator

emulator = GPEmulator.from_file(sys.argv[1])
emulator.train_emulators(max_training_iter=5000)
emulator.save()
