import numpy as np
from typing import Tuple, List

def build_filter(nelx: int, nely: int, rmin: float) -> Tuple[List[List[Tuple[int, float]]], np.ndarray]:
    """
    Precomputes the density filter weights based on element distances.

    Parameters
    ----------
    nelx, nely : int
        Number of elements in x and y directions.
    rmin : float
        Filter radius.

    Returns
    -------
    Tuple[List[List[Tuple[int, float]]], np.ndarray]
        H: Adjacency list storing (column_index, weight) for each element.
        Hs: Array of weight sums for normalization.
    """
    if nelx <= 0 or nely <= 0 or rmin <= 0:
        raise ValueError("Grid dimensions and filter radius must be positive.")

    n = nelx * nely
    H = [[] for _ in range(n)]
    Hs = np.zeros(n)

    for i in range(nelx):
        for j in range(nely):
            row = j + i * nely
            for k in range(max(i - int(rmin), 0), min(i + int(rmin) + 1, nelx)):
                for l in range(max(j - int(rmin), 0), min(j + int(rmin) + 1, nely)):
                    col = l + k * nely
                    dist = np.sqrt((i - k)**2 + (j - l)**2)
                    weight = max(0.0, rmin - dist)

                    if weight > 0:
                        H[row].append((col, weight))
                        Hs[row] += weight

    return H, Hs

def apply_filter(x: np.ndarray, H: List[List[Tuple[int, float]]], Hs: np.ndarray) -> np.ndarray:
    """
    Applies the precomputed density filter to a design variable field.

    Parameters
    ----------
    x : np.ndarray
        Raw design variables.
    H : List[List[Tuple[int, float]]]
        Precomputed weight list.
    Hs : np.ndarray
        Precomputed weight sums.

    Returns
    -------
    np.ndarray
        Filtered physical density field.
    """
    x_filtered = np.zeros_like(x)
    for i in range(len(x)):
        sum_val = 0.0
        for (j, w) in H[i]:
            sum_val += w * x[j]
        x_filtered[i] = sum_val / Hs[i]
    return x_filtered