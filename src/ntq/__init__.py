"""NTQ: Negative Thermal Quenching Analysis Toolkit

Temperature-dependent PL/EL spectral analysis using Marcus-Levich-Jortner 
formalism and Bayesian MCMC methods.
"""

__version__ = "0.1.0"
__author__ = "Kizamuel"

from . import config_utils
from . import covariance_utils
from . import fit_el_utils
from . import fit_pl_utils
from . import generate_data_utils
from . import plot_utils
from . import model_function

__all__ = [
    "config_utils",
    "covariance_utils",
    "fit_el_utils",
    "fit_pl_utils",
    "generate_data_utils",
    "plot_utils",
    "model_function",
]

try:
    from . import Emcee_utils
    __all__.append("Emcee_utils")
except ImportError:
    pass

try:
    from . import Exp_data_utils
    __all__.append("Exp_data_utils")
except ImportError:
    pass
