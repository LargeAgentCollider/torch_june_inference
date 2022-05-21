from abc import ABC
import yaml

from torch_june import TorchJune


class InferenceEngine(ABC):
    @classmethod
    def from_file(cls, fpath):
        with open(fpath, "r") as f:
            params = yaml.safe_load(f)
        return cls.from_parameters(params)

    @classmethod
    def from_parameters(cls, params):
        return cls(**params)

    def initialize_june_model(self, config_file):
        model = TorchJune.from_file(config_file)
        return model

    @staticmethod
    def read_results(self, path):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError
