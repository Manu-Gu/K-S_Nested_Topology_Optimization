"""
Evaluates the critical threshold (eps_c) against the aggregation parameter (kappa)
for the flat union-set control strictly under fixed centers.
Forms the baseline curve for mechanism ablation.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.topopt.problem import TopOptProblem
from src.topopt.ks_layers import flat_three_layer_G
from src.topopt.feasibility import find_eps_c
from experiments.utils import ensure_dirs, save_data, save_figure

def main() -> None:
    ensure_dirs()
    nelx, nely = 20, 10
    rmin = 1.5
    rho_list = [5, 8, 10, 15, 20, 30, 40, 60, 80, 100, 150]

    print(f"\n=== Build baseline & fixed centers at rmin = {rmin} ===")
    problem = TopOptProblem(nelx, nely, volfrac=0.5, penal=3.0, rmin=rmin)
    x_base = problem.baseline()
    centers_ref = problem.get_three_centers(x_base)
    print("Fixed centers_ref =", centers_ref)

    eps_list = []
    for rho in rho_list:
        print(f"\n=== Flat (fixed centers) | rho = {rho} ===")
        def G_func(x: np.ndarray, rho_val: float) -> float:
            return flat_three_layer_G(
                x, centers_ref, rho_val, problem.H, problem.Hs, nelx, nely
            )

        eps_c = find_eps_c(problem, G_func, rho, x_base)
        print(f"epsilon_c = {eps_c:.4f}")
        eps_list.append(eps_c)

    eps_list = np.array(eps_list)
    save_data(
        "flat_vs_rho_fixed_centers",
        rho=np.array(rho_list, dtype=float),
        eps_c=eps_list,
        rmin=np.array([rmin], dtype=float),
        centers_ref=np.array(centers_ref, dtype=int),
    )

    plt.figure(figsize=(6, 5))
    plt.plot(rho_list, eps_list, "o-")
    plt.xlabel(r"Aggregation parameter $\kappa$")
    plt.ylabel(r"Critical threshold $\varepsilon_c$")
    plt.title(f"Flat control (fixed centers): epsilon_c vs rho (rmin={rmin})")
    plt.grid(True)
    save_figure("flat_vs_rho_fixed_centers")
    plt.close()

if __name__ == "__main__":
    main()