import numpy as np
from scipy.optimize import minimize
from typing import List, Tuple
from .fem import assemble, solve_system, lk
from .filter import build_filter, apply_filter

class TopOptProblem:
    """
    Base class for 2D Density-based Topology Optimization problems.
    Implements standard Solid Isotropic Material with Penalization (SIMP)
    method alongside density filtering for compliance minimization.
    """

    def __init__(self, nelx: int, nely: int, volfrac: float = 0.5, penal: float = 3.0, rmin: float = 1.5):
        self.nelx = nelx
        self.nely = nely
        self.volfrac = volfrac
        self.penal = penal
        self.rmin = rmin

        self.KE = lk()
        self.H, self.Hs = build_filter(nelx, nely, rmin)
        self.ndof = 2 * (nelx + 1) * (nely + 1)

        # load vector (cantilever tip load)
        self.F = np.zeros(self.ndof)
        self.F[2 * (nely + 1) * nelx + 1] = -1.0

        # fixed DOFs (left boundary)
        self.fixed = np.union1d(
            np.arange(0, 2 * (nely + 1), 2),
            np.arange(0, 2 * (nely + 1), 2) + 1
        )

    def compliance(self, x: np.ndarray) -> float:
        """Calculates the structural compliance (strain energy) objective."""
        x_phys = apply_filter(x, self.H, self.Hs)
        K = assemble(self.nelx, self.nely, x_phys, self.penal)
        U = solve_system(K, self.F, self.fixed)
        return float(self.F @ U)

    def baseline(self) -> np.ndarray:
        """Computes the unconstrained (volume-only) baseline topology."""
        x0 = self.volfrac * np.ones(self.nelx * self.nely)
        cons = [{'type': 'ineq', 'fun': lambda x: self.volfrac - np.mean(x)}]

        result = minimize(
            self.compliance, x0, method='SLSQP',
            bounds=[(0.001, 1.0)] * (self.nelx * self.nely),
            constraints=cons, options={'maxiter': 80}
        )
        return result.x

    def _element_energy(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates element-wise unpenalized strain energy density."""
        x_phys = apply_filter(x, self.H, self.Hs)
        K = assemble(self.nelx, self.nely, x_phys, self.penal)
        U = solve_system(K, self.F, self.fixed)

        energy = np.zeros(self.nelx * self.nely)
        for elx in range(self.nelx):
            for ely in range(self.nely):
                el = ely + elx * self.nely
                n1, n2 = (self.nely + 1) * elx + ely, (self.nely + 1) * (elx + 1) + ely
                edof = np.array([2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n2+2, 2*n2+3, 2*n1+2, 2*n1+3])
                Ue = U[edof]
                energy[el] = Ue @ self.KE @ Ue

        return energy, x_phys

# --------------------------------------------------
    # Single-layer monitoring region (3x3 block)
    # --------------------------------------------------
    def get_single_region(self, x: np.ndarray) -> List[int]:
        """Identifies the 3x3 block containing the highest strain energy."""
        energy, x_phys = self._element_energy(x)

        score = energy * x_phys  # weighted energy
        margin = 2

        candidates = []

        for elx in range(margin, self.nelx - margin):
            for ely in range(margin, self.nely - margin):
                el = ely + elx * self.nely
                candidates.append((score[el], el))

        _, center_el = max(candidates)

        cx = center_el // self.nely
        cy = center_el % self.nely

        region = []

        for i in range(-1, 2):
            for j in range(-1, 2):
                ex = cx + i
                ey = cy + j
                if 0 <= ex < self.nelx and 0 <= ey < self.nely:
                    region.append(ey + ex * self.nely)

        return region

    # --------------------------------------------------
    # Three-layer centers (top-3 energy locations)
    # --------------------------------------------------
    def get_three_centers(self, x: np.ndarray) -> List[Tuple[int, int]]:
        """Identifies the top 3 high-energy centers for multi-layer tracking."""
        energy, x_phys = self._element_energy(x)

        score = energy * x_phys
        margin = 2

        candidates = []

        for elx in range(margin, self.nelx - margin):
            for ely in range(margin, self.nely - margin):
                el = ely + elx * self.nely
                candidates.append((score[el], el))

        candidates.sort(reverse=True)

        centers = []

        for k in range(3):
            el = candidates[k][1]
            cx = el // self.nely
            cy = el % self.nely
            centers.append((cx, cy))

        return centers