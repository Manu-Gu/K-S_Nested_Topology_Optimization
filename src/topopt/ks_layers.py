import numpy as np
from typing import List, Tuple, Optional
from src.ks.ks_functions import ks_aggregate
from .filter import apply_filter

def single_layer_G(x: np.ndarray, region: List[int], rho: float, H: list, Hs: np.ndarray) -> float:
    """
    Evaluates a single-layer KS constraint over a fixed spatial region.
    """
    x_phys = apply_filter(x, H, Hs)
    values = [x_phys[e] for e in region]
    return ks_aggregate(values, rho)

def three_layer_G(x: np.ndarray, centers: List[Tuple[int, int]], 
                  rho0: float, rho1: float, rho2: float,
                  H: list, Hs: np.ndarray, nelx: int, nely: int,
                  offsets: Optional[List[Tuple[int, int]]] = None,
                  local_radius: int = 1, scale_factor: int = 1) -> float:
    """
    Evaluates the three-layer nested KS constraint architecture.

    Parameters
    ----------
    x : np.ndarray
        Raw design variables.
    centers : List[Tuple[int, int]]
        List of (x, y) coordinates for the fixed monitoring centers.
    rho0, rho1, rho2 : float
        KS aggregation parameters for the block, offset, and center layers.
    scale_factor : int, optional
        Scaling factor used for mesh independence verification, by default 1.

    Returns
    -------
    float
        Aggregated nested KS value.
    """
    x_phys = apply_filter(x, H, Hs)
    scaled_radius = local_radius * scale_factor
    
    if offsets is None:
        scaled_offsets = [(0, 0), (2 * scale_factor, 0), 
                          (0, 2 * scale_factor), (2 * scale_factor, 2 * scale_factor)]
    else:
        scaled_offsets = [(dx * scale_factor, dy * scale_factor) for (dx, dy) in offsets]

    G_i = []
    for (cx, cy) in centers:
        d_ij = []
        for dx, dy in scaled_offsets:
            values = []
            for i in range(-scaled_radius, scaled_radius + 1):
                for j in range(-scaled_radius, scaled_radius + 1):
                    ex, ey = cx + dx + i, cy + dy + j
                    if 0 <= ex < nelx and 0 <= ey < nely:
                        el = ey + ex * nely
                        values.append(x_phys[el])
            if values:
                d_ij.append(ks_aggregate(values, rho0))
        if d_ij:
            G_i.append(ks_aggregate(d_ij, rho1))

    return ks_aggregate(G_i, rho2)

def collect_three_layer_elements(centers: List[Tuple[int, int]], nelx: int, nely: int,
                                 offsets: Optional[List[Tuple[int, int]]] = None,
                                 local_radius: int = 1, scale_factor: int = 1) -> List[int]:
    """
    Collects unique element indices covered by the hierarchical monitoring structure.
    """
    scaled_radius = local_radius * scale_factor
    if offsets is None:
        scaled_offsets = [(0, 0), (2 * scale_factor, 0), 
                          (0, 2 * scale_factor), (2 * scale_factor, 2 * scale_factor)]
    else:
        scaled_offsets = [(dx * scale_factor, dy * scale_factor) for (dx, dy) in offsets]

    elems = set()
    for (cx, cy) in centers:
        for (dx, dy) in scaled_offsets:
            for i in range(-scaled_radius, scaled_radius + 1):
                for j in range(-scaled_radius, scaled_radius + 1):
                    ex, ey = cx + dx + i, cy + dy + j
                    if 0 <= ex < nelx and 0 <= nely:
                        elems.add(ey + ex * nely)
    return sorted(elems)

def flat_three_layer_G(x: np.ndarray, centers: List[Tuple[int, int]], rho: float, 
                       H: list, Hs: np.ndarray, nelx: int, nely: int,
                       offsets: Optional[List[Tuple[int, int]]] = None,
                       local_radius: int = 1, scale_factor: int = 1) -> float:
    """
    Evaluates the flattened control (union set) KS constraint for ablation studies.
    """
    x_phys = apply_filter(x, H, Hs)
    elems = collect_three_layer_elements(centers, nelx, nely, offsets, local_radius, scale_factor)
    values = [x_phys[e] for e in elems]
    return ks_aggregate(values, rho)