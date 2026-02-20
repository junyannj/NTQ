"""NTQ - Negative Thermal Quenching Analysis Toolkit"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_experimental_data(filepath):
    raise NotImplementedError("Please implement the data reading function")


def read_pl_data(filepath):
    raise NotImplementedError("Please implement PL data reading function")


def read_el_data(filepath):
    raise NotImplementedError("Please implement EL data reading function")


def plot_experimental_data(data, title="Experimental Data"):
    raise NotImplementedError("Please implement plotting function")


def normalize_spectra(spectra):
    raise NotImplementedError("Please implement normalization function")


def extract_peak_positions(energy, spectra):
    raise NotImplementedError("Please implement peak extraction function")
