import pytest
from pathlib import Path
import numpy as np

from torch_june_inference.utils import read_fortran_data_file
test_path = Path(__file__).parent

class TestReadFortranFile:
    @pytest.fixture(name="fpath")
    def make_file(self):
        fpath = test_path / "fortran_test.txt"
        text = "1-3\t2e-4\t3E-1"
        with open(fpath, "w") as f:
            f.write(text)
        return fpath

    def test__read_file(self, fpath):
        ff = read_fortran_data_file(fpath)
        assert np.allclose(ff, np.array([1e-3, 2e-4, 3e-1]))

