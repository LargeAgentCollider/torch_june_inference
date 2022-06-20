import numpy as np
import re

from torch_june_inference.mpi_setup import mpi_rank

def read_fortran_data_file(file):
    # count the columns in the first row of data
    number_columns = np.genfromtxt(file, max_rows=1).shape[0]

    c = lambda s: float(re.sub(r"(\d)([\+\-])(\d)", r"\1E\2\3", s.decode()))

    # actually load the content of our file
    data = np.genfromtxt(file,
        converters=dict(zip(range(number_columns), [c] * number_columns)),)
    return data

def read_device(device):
    if device == "mpi_rank":
        device = f"cuda:{mpi_rank}"
    return device