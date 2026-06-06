# Feasible-set geometry of nested KS constraints

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains the official Python source code for the paper:
**"Feasible-set geometry of nested KS constraints in topology optimization: Mechanism, quantification, and compensation"** (Submitted to *Finite Elements in Analysis and Design*).

## Overview

The Kreisselmeier–Steinhauser (KS) function is widely utilized to approximate non-smooth extremal constraints in topology optimization. This repository provides a rigorous mathematical sandbox to investigate how hierarchical multi-layer nesting of KS operators distorts the underlying feasible-set geometry.

By employing a robust bisection-based feasibility search combined with multi-start initializations (to mitigate spurious infeasibility traps), this framework quantitatively evaluates the critical feasibility threshold ($\varepsilon_c$). Furthermore, it includes the implementation of a topology-aware stratified $\kappa$ scaling strategy to successfully invert the compounded mathematical bias.

## Repository Structure

```text
K-S_Nested_Topology_Optimization/
├── src/                 # Core source code (FEM, SIMP filter, nested KS operators)
├── experiments/         # Numerical experiment scripts to generate threshold data
├── plotting/            # Publication-quality figure generation pipeline
├── results/             
│   ├── data/            # Pre-computed .npz data files
│   └── figures/         # Generated high-resolution PDFs and PNGs
├── requirements.txt     # Python dependencies
├── CITATION.cff         # Citation metadata for GitHub
└── README.md            # Project documentation
```

## Requirements & Installation

The code is built on standard scientific Python libraries and requires **Python 3.8 or higher**. To isolate the environment and install dependencies, run the following commands:

```bash
# Create and activate a virtual environment (Recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install required dependencies (NumPy, SciPy, Matplotlib)
pip install -r requirements.txt
```

## Reproducibility Guide

This repository is designed to fully reproduce the results, tables, and figures presented in the manuscript.

### Step 1: Run Numerical Experiments (Generate Data)

All experiment scripts are located in the `experiments/` directory. These scripts perform dense Finite Element Analysis (FEA) and Sequential Least Squares Programming (SLSQP) optimizations to output `.npz` data files.

**Example:** Run the ablation study comparing nested vs. flat constraints:
```bash
python experiments/exp_compare_nested_flat_rho_fixed_centers.py
```

> ⚠️ **Computational Cost Note:**
> - Standard $20 \times 10$ benchmarks (e.g., parameter scaling, filter radius sensitivity) typically converge within a few minutes on a standard desktop CPU.
> - The refined $40 \times 20$ mesh independence verification involves an $O(N^3)$ sparse solver scaling and may require 2~4 hours.

### Step 2: Generate Figures

Once the data is generated, you can recreate all the exact figures (Fig 1 through Fig 6, plus Supplementary figures) used in the paper. The plotting pipeline automatically utilizes LaTeX rendering (if available) and an advanced hash-caching mechanism to avoid redundant optimization runs.

```bash
python plotting/generate_all_figures.py
```
*Generated figures will be saved as high-resolution PDFs and PNGs in the `results/figures/` directory.*

## Citation

If you find this code or our theoretical framework useful in your research, please consider citing our paper:

```bibtex
@article{gu2026feasible,
  title={Feasible-set geometry of nested KS constraints in topology optimization: Mechanism, quantification, and compensation},
  author={Gu, Hengdong},
  journal={Finite Elements in Analysis and Design},
  year={2026},
  note={Submitted}
}
```
*(Note: This BibTeX entry will be updated with volume, issue, and DOI details upon publication.)*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions regarding the mathematical framework, implementation details, or the paper itself, please open an issue on GitHub or contact the corresponding author at: [Manu_Gu@163.com](mailto:Manu_Gu@163.com).

