import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from typing import Union

def lk(E: float = 1.0, nu: float = 0.3) -> np.ndarray:
    """
    Computes the element stiffness matrix for a 4-node quadrilateral element (plane stress).

    Parameters
    ----------
    E : float, optional
        Young's modulus, by default 1.0.
    nu : float, optional
        Poisson's ratio, by default 0.3.

    Returns
    -------
    np.ndarray
        8x8 element stiffness matrix.
    """
    k = np.array([
        1/2 - nu/6,   1/8 + nu/8,  -1/4 - nu/12, -1/8 + 3*nu/8,
       -1/4 + nu/12, -1/8 - nu/8,   nu/6,         1/8 - 3*nu/8
    ])

    KE = E / (1 - nu**2) * np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
    ])
    return KE

def assemble(nelx: int, nely: int, x: np.ndarray, penal: float, 
             E0: float = 1.0, Emin: float = 1e-9, nu: float = 0.3) -> coo_matrix:
    """
    Vectorized assembly of the global stiffness matrix for SIMP.

    Parameters
    ----------
    nelx, nely : int
        Number of elements in x and y directions.
    x : np.ndarray
        Physical density field (flattened 1D array).
    penal : float
        SIMP penalization power (p).
    E0, Emin : float, optional
        Solid and void material Young's moduli.
    nu : float, optional
        Poisson's ratio.

    Returns
    -------
    coo_matrix
        Global sparse stiffness matrix in CSC format.
    """
    KE = lk(E0, nu)
    ndof = 2 * (nelx + 1) * (nely + 1)

    elx, ely = np.meshgrid(np.arange(nelx), np.arange(nely), indexing='ij')
    elx = elx.flatten()
    ely = ely.flatten()

    n1 = (nely + 1) * elx + ely
    n2 = (nely + 1) * (elx + 1) + ely

    edofMat = np.column_stack([
        2*n1, 2*n1+1, 2*n2, 2*n2+1, 
        2*n2+2, 2*n2+3, 2*n1+2, 2*n1+3
    ])

    iK = np.repeat(edofMat, 8, axis=1).flatten()
    jK = np.tile(edofMat, (1, 8)).flatten()

    Ee = Emin + x**penal * (E0 - Emin)
    sK = (Ee[:, np.newaxis, np.newaxis] * KE[np.newaxis, :, :]).flatten()

    K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
    return K

def solve_system(K: coo_matrix, F: np.ndarray, fixed_dofs: np.ndarray) -> np.ndarray:
    """
    Solves the linear finite element equilibrium system KU = F.

    Parameters
    ----------
    K : coo_matrix or csc_matrix
        Global stiffness matrix.
    F : np.ndarray
        Global load vector.
    fixed_dofs : np.ndarray
        Array of fixed degrees of freedom indices.

    Returns
    -------
    np.ndarray
        Global displacement vector U.
    """
    ndof = K.shape[0]
    free = np.setdiff1d(np.arange(ndof), fixed_dofs)

    U = np.zeros(ndof)
    U[free] = spsolve(K[free, :][:, free], F[free])
    return U