import numpy as np
from scipy.optimize import minimize


def find_eps_c(problem,
               G_func,
               rho,
               x_base,
               eps_min=0.0,
               eps_max=None,
               tol=1e-3,
               maxiter=150,
               feas_tol=1e-4,
               return_design=False):
    """
    Robust bisection for epsilon_c using warm-start + multi-start.

    If return_design=True, return (eps_c, x_feas).
    Otherwise return eps_c only (backward compatible).
    """

    nvar = problem.nelx * problem.nely

    # Upper bound: G(x_base, rho)
    if eps_max is None:
        eps_high = float(G_func(x_base, rho))
    else:
        eps_high = float(eps_max)

    eps_low = float(eps_min)

    # last feasible design (warm start)
    x_feas = x_base.copy()

    # extra start points for multi-start
    x_uniform = problem.volfrac * np.ones(nvar)

    def solve_at_eps(eps, x0):
        def ks_constraint(x):
            return eps - G_func(x, rho)

        constraints = [
            {'type': 'ineq', 'fun': lambda x: problem.volfrac - np.mean(x)},
            {'type': 'ineq', 'fun': ks_constraint}
        ]

        res = minimize(
            problem.compliance,
            x0,
            method='SLSQP',
            bounds=[(0.001, 1.0)] * nvar,
            constraints=constraints,
            options={'maxiter': maxiter, 'ftol': 1e-6}
        )

        x_opt = res.x
        g_val = float(G_func(x_opt, rho))
        ok = (g_val <= eps + feas_tol)
        return ok, x_opt, res.success, g_val

    # bisection
    while (eps_high - eps_low) > tol:
        eps_mid = 0.5 * (eps_low + eps_high)

        feasible = False
        best_x = None

        for x0 in (x_feas, x_base, x_uniform):
            ok, x_opt, success, g_val = solve_at_eps(eps_mid, x0.copy())
            if ok:
                feasible = True
                best_x = x_opt
                break

        if feasible:
            eps_high = eps_mid
            x_feas = best_x.copy()
        else:
            eps_low = eps_mid

    if return_design:
        return eps_high, x_feas
    return eps_high