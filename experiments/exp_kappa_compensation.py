"""
Reproduces the topology-aware stratified kappa scaling strategy designed
to proactively mitigate operator-induced feasible-set contraction.

Matches Section 4.6 (Compensatory design: Stratified kappa scaling)
and Table 3 in the submitted manuscript.
"""

import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.topopt.problem import TopOptProblem
from src.topopt.feasibility import find_eps_c
from src.topopt.ks_layers import three_layer_G

class CantileverBaseline(TopOptProblem):
    """Specific 20x10 boundary condition definitions for the compensation test."""
    def __init__(self, nelx: int = 20, nely: int = 10, volfrac: float = 0.5, penal: float = 3.0, rmin: float = 1.5):
        super().__init__(nelx, nely, volfrac, penal, rmin)
        
        self.F = np.zeros(self.ndof)
        bottom_right_node = (self.nelx + 1) * (self.nely + 1) - 1
        self.F[2 * bottom_right_node + 1] = -1.0
        
        left_edge_nodes = np.arange(self.nely + 1)
        self.fixed = np.union1d(2 * left_edge_nodes, 2 * left_edge_nodes + 1)

def run_kappa_compensation() -> None:
    print("Starting Stratified Kappa Compensation Experiment (20x10)...")
    problem = CantileverBaseline(nelx=20, nely=10, rmin=1.5)
    
    print("Computing baseline structure...")
    x_base = problem.baseline()
    
    # Standard 20x10 monitoring centers
    fixed_centers = [(5,6), (5,3), (8,2)]
    
    scenarios = [
        {"name": "Nested: Uniform", "kappas": (20, 20, 20)},
        {"name": "Nested: Stratified 1", "kappas": (20, 40, 60)},
        {"name": "Nested: Stratified 2", "kappas": (20, 60, 100)}
    ]
    
    results = []
    for sc in scenarios:
        k1, k2, k3 = sc["kappas"]
        name = sc["name"]
        print(f"\n--- Testing Scenario: {name} | k1={k1}, k2={k2}, k3={k3} ---")
        
        def G_stratified_wrapper(x: np.ndarray, _: float) -> float: 
            return three_layer_G(x, fixed_centers, k1, k2, k3, 
                                 problem.H, problem.Hs, problem.nelx, problem.nely)
        
        eps_c = find_eps_c(problem, G_stratified_wrapper, 20, x_base, eps_max=1.5, tol=1e-3)
        print(f"Result for {name}: eps_c = {eps_c:.4f}")
        results.append((name, k1, k2, k3, eps_c))

    print("\n================ FINAL RESULTS (MATCHES TABLE 3) ================")
    print(f"{'Strategy':<25} | {'Configuration':<30} | {'Threshold (eps_c)'}")
    print("-" * 75)
    print(f"{'Single-layer Baseline':<25} | {'kappa = 20 (Flat reference)':<30} | 0.1111")
    for name, k1, k2, k3, eps in results:
        config_str = f"k1={k1}, k2={k2}, k3={k3}"
        print(f"{name:<25} | {config_str:<30} | {eps:.4f}")

if __name__ == "__main__":
    run_kappa_compensation()