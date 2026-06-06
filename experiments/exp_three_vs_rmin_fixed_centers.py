"""
Evaluates the sensitivity of eps_c against filter radius (rmin) for the three-layer 
nested KS constraint, explicitly exposing the smoothness-bias amplification.

Matches Section 4.3, Table 2, and Figure 4 in the submitted manuscript.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.topopt.problem import TopOptProblem
from src.topopt.ks_layers import three_layer_G
from src.topopt.feasibility import find_eps_c
from experiments.utils import ensure_dirs, save_data, save_figure

def main() -> None:
    ensure_dirs()
    nelx, nely = 20, 10
    rho = 20.0
    rmin_list = [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6]
    rmin_ref = 1.5

    print(f"\n=== Build fixed centers at rmin_ref = {rmin_ref} ===")
    problem_ref = TopOptProblem(nelx, nely, volfrac=0.5, penal=3.0, rmin=rmin_ref)
    x_base_ref = problem_ref.baseline()
    centers_ref = problem_ref.get_three_centers(x_base_ref)
    print("Fixed centers_ref =", centers_ref)

    eps_list = []
    for rmin in rmin_list:
        print(f"\n=== Three-layer (fixed centers) | rmin = {rmin} ===")
        problem = TopOptProblem(nelx, nely, volfrac=0.5, penal=3.0, rmin=rmin)
        x_base = problem.baseline()

        def G_func(x: np.ndarray, rho_val: float) -> float:
            return three_layer_G(
                x, centers_ref, rho_val, rho_val, rho_val,
                problem.H, problem.Hs, nelx, nely
            )

        eps_c = find_eps_c(problem, G_func, rho, x_base)
        print(f"epsilon_c = {eps_c:.4f}")
        eps_list.append(eps_c)

    eps_list = np.array(eps_list)

    save_data(
        "three_vs_rmin_fixed_centers",
        rmin=np.array(rmin_list, dtype=float),
        eps_c=eps_list,
        rho=np.array([rho], dtype=float),
        rmin_ref=np.array([rmin_ref], dtype=float),
        centers_ref=np.array(centers_ref, dtype=int),
    )

    plt.figure(figsize=(6, 5))
    plt.plot(rmin_list, eps_list, "o-")
    plt.xlabel(r"Filter radius $r_{\min}$")
    plt.ylabel(r"Critical threshold $\varepsilon_c$")
    plt.title(f"Three-layer (fixed centers): epsilon_c vs r_min (rho={rho}, rmin_ref={rmin_ref})")
    plt.grid(True)
    save_figure("three_vs_rmin_fixed_centers")
    plt.close()

if __name__ == "__main__":
    main()