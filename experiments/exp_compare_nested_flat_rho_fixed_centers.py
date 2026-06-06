"""
Reproduces the geometric ablation study evaluating the critical threshold (eps_c)
against the aggregation parameter (rho / kappa) under fixed monitoring centers.

Matches Section 4.4 (Geometric ablation: nested vs. flat union-set) 
and Figure 5a in the submitted manuscript.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure the 'src' module can be imported regardless of execution directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.topopt.problem import TopOptProblem
from src.topopt.ks_layers import three_layer_G, flat_three_layer_G
from src.topopt.feasibility import find_eps_c
from experiments.utils import ensure_dirs, save_data, save_figure

def main() -> None:
    ensure_dirs()

    scale_factor = 3  
    base_nelx, base_nely = 20, 10
    nelx = base_nelx * scale_factor
    nely = base_nely * scale_factor

    base_rmin = 1.5
    rmin = base_rmin * scale_factor

    rho_list = [5, 8, 10, 15, 20, 30, 40, 60, 80, 100, 150]

    print(f"Initializing Problem: Grid {nelx}x{nely}, rmin={rmin}, scale={scale_factor}")
    problem = TopOptProblem(nelx, nely, volfrac=0.5, penal=3.0, rmin=rmin)
    
    x_base = problem.baseline()
    centers_ref = problem.get_three_centers(x_base)
    print("Fixed centers_ref =", centers_ref)

    eps_nested, eps_flat = [], []

    for rho in rho_list:
        print(f"\n=== scale = {scale_factor} | rho = {rho} ===")

        def G_nested(x: np.ndarray, rho_val: float) -> float:
            return three_layer_G(
                x, centers_ref, rho_val, rho_val, rho_val,
                problem.H, problem.Hs, nelx, nely, scale_factor=scale_factor 
            )

        def G_flat(x: np.ndarray, rho_val: float) -> float:
            return flat_three_layer_G(
                x, centers_ref, rho_val,
                problem.H, problem.Hs, nelx, nely, scale_factor=scale_factor 
            )

        e1 = find_eps_c(problem, G_nested, rho, x_base)
        e2 = find_eps_c(problem, G_flat, rho, x_base)

        print(f"eps_nested = {e1:.4f} | eps_flat = {e2:.4f}")
        eps_nested.append(e1)
        eps_flat.append(e2)

    eps_nested = np.array(eps_nested)
    eps_flat = np.array(eps_flat)

    file_prefix = f"compare_nested_flat_vs_rho_scale{scale_factor}"

    save_data(
        file_prefix,
        rho=np.array(rho_list, dtype=float),
        eps_nested=eps_nested,
        eps_flat=eps_flat,
        rmin=np.array([rmin], dtype=float),
        centers_ref=np.array(centers_ref, dtype=int),
        scale_factor=np.array([scale_factor], dtype=int)
    )

    plt.figure(figsize=(6, 5))
    plt.plot(rho_list, eps_nested, "o-", label="Nested three-layer")
    plt.plot(rho_list, eps_flat, "s--", label="Flat (union set)")
    plt.xlabel(r"Aggregation parameter $\kappa$")
    plt.ylabel(r"Critical threshold $\varepsilon_c$")
    plt.title(f"Nested vs Flat (Grid {nelx}x{nely}), rmin={rmin}")
    plt.grid(True)
    plt.legend()

    save_figure(file_prefix)
    plt.close()

if __name__ == "__main__":
    main()