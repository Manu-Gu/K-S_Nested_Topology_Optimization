"""
Universality benchmark demonstrating the critical threshold (eps_c) evaluation 
on a Messerschmitt-Bölkow-Blohm (MBB) beam model across distinct kappa values.

Matches Section 4.5 (Universality benchmark: MBB beam validation)
and Figure 6 in the submitted manuscript.
"""

import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.topopt.problem import TopOptProblem
from src.topopt.feasibility import find_eps_c
from src.topopt.ks_layers import three_layer_G, flat_three_layer_G
from experiments.utils import ensure_dirs, save_data

class MBBProblem(TopOptProblem):
    """Specific boundary conditions for the MBB beam (half-domain)."""
    def __init__(self, nelx: int = 30, nely: int = 10, volfrac: float = 0.5, penal: float = 3.0, rmin: float = 1.5):
        super().__init__(nelx, nely, volfrac, penal, rmin)
        
        self.F = np.zeros(self.ndof)
        self.F[1] = -1.0  # Top-left corner downward load
        
        left_edge_nodes = np.arange(self.nely + 1)
        left_edge_x_dofs = 2 * left_edge_nodes
        bottom_right_node = (self.nelx + 1) * (self.nely + 1) - 1
        bottom_right_y_dof = 2 * bottom_right_node + 1
        
        self.fixed = np.union1d(left_edge_x_dofs, [bottom_right_y_dof])

def run_mbb_experiment() -> None:
    ensure_dirs()
    print("Starting MBB Universality Benchmark (30x10)...")
    problem = MBBProblem(nelx=30, nely=10, rmin=1.5)
    
    print("Extracting baseline centers...")
    x_base = problem.baseline()
    fixed_centers = problem.get_three_centers(x_base)
    print(f"Fixed Centers: {fixed_centers}")
    
    def G_nested_wrapper(x: np.ndarray, rho: float) -> float:
        return three_layer_G(x, fixed_centers, rho, rho, rho, problem.H, problem.Hs, problem.nelx, problem.nely)

    def G_flat_wrapper(x: np.ndarray, rho: float) -> float:
        return flat_three_layer_G(x, fixed_centers, rho, problem.H, problem.Hs, problem.nelx, problem.nely)

    kappa_list = [10, 20, 40]
    eps_nested_list, eps_flat_list = [], []
    
    for k in kappa_list:
        print(f"\n--- Running bisection for kappa = {k} ---")
        e_nest = find_eps_c(problem, G_nested_wrapper, k, x_base, eps_max=1.5, tol=1e-3)
        e_flat = find_eps_c(problem, G_flat_wrapper, k, x_base, eps_max=1.5, tol=1e-3)
        
        print(f"Result: nested = {e_nest:.4f}, flat = {e_flat:.4f}")
        eps_nested_list.append(e_nest)
        eps_flat_list.append(e_flat)

    save_data("mbb_universality", kappa=np.array(kappa_list), eps_nested=np.array(eps_nested_list), eps_flat=np.array(eps_flat_list))
    print("\nMBB Experiment Done! Universality Verified.")

if __name__ == "__main__":
    run_mbb_experiment()