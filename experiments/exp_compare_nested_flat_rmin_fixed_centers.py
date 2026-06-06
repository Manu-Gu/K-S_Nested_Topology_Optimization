"""
Reproduces the geometric ablation study evaluating the critical threshold (eps_c)
against the filter radius (rmin) comparing nested vs. flat KS constraints.

Matches Section 4.4 (Geometric ablation) and Figure 5b in the submitted manuscript.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.topopt.problem import TopOptProblem
from src.topopt.ks_layers import three_layer_G, flat_three_layer_G
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

    eps_nested, eps_flat = [], []

    for rmin in rmin_list:
        print(f"\n=== Compare nested vs flat | rmin = {rmin} ===")
        problem = TopOptProblem(nelx, nely, volfrac=0.5, penal=3.0, rmin=rmin)
        x_base = problem.baseline()

        def G_nested(x: np.ndarray, rho_val: float) -> float:
            return three_layer_G(
                x, centers_ref, rho_val, rho_val, rho_val,
                problem.H, problem.Hs, nelx, nely
            )

        def G_flat(x: np.ndarray, rho_val: float) -> float:
            return flat_three_layer_G(
                x, centers_ref, rho_val,
                problem.H, problem.Hs, nelx, nely
            )

        e_nested = find_eps_c(problem, G_nested, rho, x_base)
        e_flat = find_eps_c(problem, G_flat, rho, x_base)

        print(f"eps_nested = {e_nested:.4f} | eps_flat = {e_flat:.4f}")
        eps_nested.append(e_nested)
        eps_flat.append(e_flat)

    eps_nested = np.array(eps_nested)
    eps_flat = np.array(eps_flat)

    save_data(
        "compare_nested_flat_vs_rmin_fixed_centers",
        rmin=np.array(rmin_list, dtype=float),
        eps_nested=eps_nested,
        eps_flat=eps_flat,
        rho=np.array([rho], dtype=float),
        rmin_ref=np.array([rmin_ref], dtype=float),
        centers_ref=np.array(centers_ref, dtype=int),
    )

    plt.figure(figsize=(6, 5))
    plt.plot(rmin_list, eps_nested, "o-", label="Nested three-layer")
    plt.plot(rmin_list, eps_flat, "s--", label="Flat (union set)")
    plt.xlabel(r"Filter radius $r_{\min}$")
    plt.ylabel(r"Critical threshold $\varepsilon_c$")
    plt.title(f"Nested vs Flat (fixed centers), rho={rho}, rmin_ref={rmin_ref}")
    plt.grid(True)
    plt.legend()

    save_figure("compare_nested_flat_vs_rmin_fixed_centers")
    plt.close()

if __name__ == "__main__":
    main()