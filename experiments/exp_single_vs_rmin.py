"""
Evaluates the sensitivity of eps_c against filter radius (rmin) for a single-layer 
KS constraint. Demonstrates the robustness of single-layer controls to smoothing changes.

Matches Section 4.3, Table 2, and Figure 4 in the submitted manuscript.
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
    rho = 20.0
    rmin_list = [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6]
    eps_list = []

    for rmin in rmin_list:
        print(f"\n=== Single-layer | rmin = {rmin} ===")
        problem = TopOptProblem(nelx, nely, volfrac=0.5, penal=3.0, rmin=rmin)
        x_base = problem.baseline()
        region = problem.get_single_region(x_base)

        def G_func(x: np.ndarray, rho_val: float) -> float:
            return single_layer_G(x, region, rho_val, problem.H, problem.Hs)

        eps_c = find_eps_c(problem, G_func, rho, x_base)
        print(f"epsilon_c = {eps_c:.4f}")
        eps_list.append(eps_c)

    eps_list = np.array(eps_list)
    save_data("single_vs_rmin", rmin=np.array(rmin_list), eps_c=eps_list)

    plt.figure(figsize=(6, 5))
    plt.plot(rmin_list, eps_list, 'o-')
    plt.xlabel(r"Filter radius $r_{\min}$")
    plt.ylabel(r"Critical threshold $\varepsilon_c$")
    plt.title("Single-layer: epsilon_c vs r_min")
    plt.grid(True)
    save_figure("single_vs_rmin")
    plt.close()

if __name__ == "__main__":
    main()