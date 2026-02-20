# NTQ

Negative Thermal Quenching Analysis Toolkit

A Python package for analyzing temperature-dependent photoluminescence (PL) and electroluminescence (EL) spectra to observe and analyze negative thermal quenching phenomena. Uses Marcus-Levich-Jortner formalism and Bayesian MCMC methods.

## Installation

```bash
git clone https://github.com/junyannj/NTQ.git
cd NTQ
pip install -e .
```

## Quick Start

```python
from ntq import fit_pl_utils, fit_el_utils
from ntq import Exp_data_utils, plot_utils

# Load data
data = Exp_data_utils.read_experimental_data("your_data.csv")

# Run fitting
sampler = fit_pl_utils.run_PL_sampling(data, config)

# Visualize
plot_utils.plot_corner(sampler)
```

## Features

- Temperature-dependent PL/EL spectral fitting
- Negative thermal quenching (NTQ) analysis
- Bayesian MCMC fitting with emcee
- Marcus-Levich-Jortner formalism
- Exciton and charge-transfer state modeling
- Covariance analysis
- Visualization tools

## Citation

```bibtex
@software{ntq,
  title={NTQ: Negative Thermal Quenching Analysis Toolkit},
  author={Kizamuel},
  year={2026},
  url={https://github.com/junyannj/NTQ}
}
```

## Acknowledgments

Based on DriftFusionOPV by Mohammed Azzouzi ([@mohammedazzouzi15](https://github.com/mohammedazzouzi15)) and the Jenny Nelson Group.

```bibtex
@software{driftfusionopv,
  title={DriftFusionOPV},
  author={Azzouzi, Mohammed},
  year={2022},
  url={https://github.com/Jenny-Nelson-Group/DriftFusionOPV}
}
```

## License

MIT License - see LICENSE file
