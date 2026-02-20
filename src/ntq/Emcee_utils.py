"""NTQ - Negative Thermal Quenching Analysis Toolkit"""

import h5py
import numpy as np
from emcee import backends, EnsembleSampler


class CustomHDF5Backend(backends.HDFBackend):
    """Custom HDF5 backend for emcee sampler."""
    
    def __init__(self, filename, name="mcmc", **kwargs):
        super().__init__(filename, name=name, **kwargs)
    
    def save_metadata(self, metadata):
        pass
    
    def load_metadata(self):
        return {}


hDFBackend_2 = backends.HDFBackend
ensemble_sampler = EnsembleSampler


def setup_backend(filename, n_dim, n_walkers, reset=False):
    backend = CustomHDF5Backend(filename)
    if reset:
        backend.reset(n_walkers, n_dim)
    return backend


def get_autocorr_time(sampler, quiet=False):
    try:
        tau = sampler.get_autocorr_time(quiet=quiet)
        return tau
    except Exception as e:
        if not quiet:
            print(f"Could not calculate autocorrelation time: {e}")
        return None


def check_convergence(sampler, min_samples=50):
    tau = get_autocorr_time(sampler, quiet=True)
    if tau is None:
        return False
    n_steps = sampler.iteration
    converged = np.all(n_steps > min_samples * tau)
    return converged


def save_chain(sampler, filename):
    raise NotImplementedError("Please implement chain saving function")


def load_chain(filename):
    raise NotImplementedError("Please implement chain loading function")
