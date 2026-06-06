import numpy as np

def ks_aggregate(values: np.ndarray, rho: float) -> float:
    """
    Kreisselmeier–Steinhauser (KS) aggregation function.

    Computes a smooth, differentiable approximation to the maximum
    of a set of values. Commonly used in structural optimization
    and constrained aggregation.

    Reference
    ---------
    Kreisselmeier, G., & Steinhauser, R. (1979). Systematic control
    design by optimizing a vector performance index. IFAC Proceedings.

    Parameters
    ----------
    values : np.ndarray or array-like
        Input values to aggregate. Must be non-empty.
    rho : float
        Aggregation parameter. Must be > 0.
        Larger rho -> closer approximation to max(values).

    Returns
    -------
    float
        KS aggregated scalar value.

    Raises
    ------
    ValueError
        If `values` is empty or `rho` <= 0.
    """
    values = np.asarray(values)

    if values.size == 0:
        raise ValueError("ks_aggregate received empty input.")
    
    if rho <= 0:
        raise ValueError(f"Aggregation parameter rho must be > 0, got {rho}")

    vmax = np.max(values)

    return vmax + (1.0 / rho) * np.log(
        np.sum(np.exp(rho * (values - vmax)))
    )