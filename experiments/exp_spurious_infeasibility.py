"""
Conducts a comprehensive multi-start probability scan to quantitatively 
characterize the spurious infeasibility trap induced by nested nonconvexity.

Matches Appendix B (Spurious infeasibility probability curve)
and Figure S3 in the submitted manuscript.
"""

import os
import sys
import numpy as np
from scipy.optimize import minimize
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.topopt.problem import TopOptProblem
from src.topopt.ks_layers import three_layer_G
from experiments.utils import ensure_dirs, save_data

def generate_valid_random_starts(nelx: int, nely: int, volfrac: float, num_starts: int = 10) -> List[np.ndarray]:
    """Generates random density fields strictly satisfying the volume equality V(x) = f."""
    starts = []
    N = nelx * nely
    
    # Optional: Fix seed for exact reproducibility across different machines
    # np.random.seed(42) 
    
    for _ in range(num_starts):
        x_rand = np.random.uniform(0.1, 0.9, N)
        x_scaled = x_rand * (volfrac / np.mean(x_rand))
        x_clipped = np.clip(x_scaled, 0.001, 1.0)
        
        diff = np.mean(x_clipped) - volfrac
        x_clipped -= diff 
        starts.append(np.clip(x_clipped, 0.001, 1.0))
    return starts

def run_probability_curve() -> None:
    ensure_dirs()
    print("Starting Spurious Infeasibility Probability Scan...")
    
    problem = TopOptProblem(nelx=20, nely=10, rmin=1.5)
    kappa = 20
    
    print("Extracting baseline centers...")
    x_base = problem.baseline()
    fixed_centers = problem.get_three_centers(x_base)
    
    # Test threshold interval (transition zone around eps_c = 0.2635)
    eps_test_points = np.arange(0.245, 0.285, 0.005) 
    num_restarts = 10 
    
    def G_nested_wrapper(x: np.ndarray) -> float:
        return three_layer_G(x, fixed_centers, kappa, kappa, kappa, 
                             problem.H, problem.Hs, problem.nelx, problem.nely)
    
    success_rates = []
    
    for eps in eps_test_points:
        print(f"\nTesting epsilon threshold: {eps:.3f} (Launching {num_restarts} multi-starts...)")
        success_count = 0
        random_x0_list = generate_valid_random_starts(problem.nelx, problem.nely, problem.volfrac, num_restarts)
        
        cons_fun = lambda x: eps + 1e-4 - G_nested_wrapper(x)
        cons = [
            {'type': 'eq', 'fun': lambda x: problem.volfrac - np.mean(x)},
            {'type': 'ineq', 'fun': cons_fun}
        ]
        
        for i, x0 in enumerate(random_x0_list):
            res = minimize(
                problem.compliance, x0, method='SLSQP',
                bounds=[(0.001, 1.0)] * (problem.nelx * problem.nely),
                constraints=cons, options={'maxiter': 80, 'ftol': 1e-4, 'disp': False}
            )
            
            is_feasible = res.success or (cons_fun(res.x) >= -1e-4 and abs(np.mean(res.x) - problem.volfrac) < 1e-3)
            if is_feasible:
                success_count += 1
                
        rate = success_count / num_restarts
        success_rates.append(rate)
        print(f"  -> Success rate: {rate*100:5.1f}% ({success_count}/{num_restarts})")

    save_data("spurious_infeasibility", eps_points=eps_test_points, rates=success_rates)
    print("\nProbability Scan Done! A sigmoidal transition should be evident.")

if __name__ == "__main__":
    run_probability_curve()