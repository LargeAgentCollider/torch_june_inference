import pytest
import torch
import numpy as np

from torch_june_inference.emulation import GPEmulator

class TestGPEmulation:
    @pytest.fixture(name="data")
    def make_data(self):
        xs = torch.tensor([[0,0,0.], [1,1,1.]])
        ys = 2 * xs + 1
        return xs, ys

    @pytest.fixture(name="emulator")
    def build_emulator(self, data):
        xs, ys = data
        return GPEmulator(train_x=xs, train_y=ys)

    def test__emulation(self, data, emulator):
        emulator.train_emulator(max_training_iter=10)
        emulator.set_eval()
        xs, ys = data
        # test inputs
        with torch.no_grad():
            for i in range(2):
                res = emulator(xs[i].reshape(1,-1))
                mean = res.mean
                lower, upper = res.confidence_region()
                assert (lower < ys[i]).all()
                assert (upper > ys[i]).all()

        # test interpolation
        xs = torch.tensor([[0.3, 0.2, 0.4], [0.5, 0.6, 0.7]])
        ys = 2 * xs + 1
        with torch.no_grad():
            for i in range(2):
                res = emulator(xs[i].reshape(1,-1))
                mean = res.mean
                lower, upper = res.confidence_region()
                assert (lower < ys[i]).all()
                assert (upper > ys[i]).all()
