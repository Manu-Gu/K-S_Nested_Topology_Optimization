"""
Evaluates the geometric response of eps_c against kappa for a single-layer KS constraint.
Provides the fundamental baseline for multi-layer nesting comparisons.

Matches Section 4.2, Table 1, and Figure 3 in the submitted manuscript.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.topopt.problem import TopOptProblem
from src.topopt.ks_layers import single_layer_G
from src.topopt.feasibility import find_eps_c
from experiments.utils import ensure_dirs, save_data, save_figure

def main() -> None:
    ensure_dirs()
    nelx, nely = 20, 10
    rmin = 1.5
    rho_list = [5, 8, 10, 15, 20, 30, 40, 60, 80, 100, 150]
    eps_list = []

    problem = TopOptProblem(nelx, nely, volfrac=0.5, penal=3.0, rmin=rmin)
    x_base = problem.baseline()
    region = problem.get_single_region(x_base)

    for rho in rho_list:
        print(f"\n=== Single-layer | rho = {rho} ===")
        def G_func(x: np.ndarray, rho_val: float) -> float:
            return single_layer_G(x, region, rho_val, problem.H, problem.Hs)

        eps_c = find_eps_c(problem, G_func, rho, x_base)
        print(f"epsilon_c = {eps_c:.4f}")
        eps_list.append(eps_c)

    eps_list = np.array(eps_list)
    save_data("single_vs_rho", rho=np.array(rho_list), eps_c=eps_list)

    plt.figure(figsize=(6, 5))
    plt.plot(rho_list, eps_list, 'o-')
    plt.xlabel(r"Aggregation parameter $\kappa$")
    plt.ylabel(r"Critical threshold $\varepsilon_c$")
    plt.title("Single-layer: epsilon_c vs rho")
    plt.grid(True)
    save_figure("single_vs_rho")
    plt.close()

if __name__ == "__main__":
    main()