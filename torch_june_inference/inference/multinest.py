from pathlib import Path
import pymultinest
import numpy as np

from torch_june_inference.utils import read_fortran_data_file


class MultiNest:
    def __init__(self, model, prior, loglike, ndim, output_path="./multinest"):
        self.model = model
        self.prior = prior
        self.loglike = loglike
        self.ndim = ndim
        self.output_path = self._read_path(output_path)
        self.samples_weights = None
        self.likelihood_values = None
        self.samples = None

    def _read_path(self, output_path):
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True, parents=True)
        output_path = output_path / "multinest"
        return output_path.as_posix()

    def run(self, x_obs, y_obs, **kwargs):
        def _loglike(cube, ndim, nparams):
            y = self.model(param=cube, x=x_obs)
            return self.loglike(y=y, y_obs=y_obs)

        pymultinest.run(
            _loglike,
            self.prior,
            self.ndim,
            outputfiles_basename=self.output_path,
            **kwargs
        )
        results = self.read_results(self.output_path)
        self.samples_weights = results[:, 0]
        self.likelihood_values = results[:, 1]
        self.samples = results[:, 2:]

    @staticmethod
    def read_results(fpath):
        results = read_fortran_data_file(fpath + ".txt")
        return results
