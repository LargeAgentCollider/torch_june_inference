from torch_june_inference.emulation import GPEmulator

emulator = GPEmulator.from_file("./configs/emulator.yaml")
emulator.train_emulator(max_training_iter=1000)
emulator.save()
