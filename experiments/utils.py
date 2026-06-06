import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Any

def ensure_dirs() -> None:
    """
    Ensures that the output directories for results and figures exist.
    Called at the beginning of experimental scripts.
    """
    os.makedirs("results/data", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

def save_data(filename: str, **kwargs: Any) -> None:
    """
    Saves experimental numerical data to a compressed .npz file.
    
    Parameters
    ----------
    filename : str
        The base name of the file (without extension).
    **kwargs : Any
        Key-value pairs of numpy arrays to save.
    """
    np.savez(f"results/data/{filename}.npz", **kwargs)

def save_figure(filename: str) -> None:
    """
    Saves the current matplotlib figure to a high-resolution PNG.
    
    Parameters
    ----------
    filename : str
        The base name of the image (without extension).
    """
    plt.tight_layout()
    plt.savefig(f"results/figures/{filename}.png", dpi=300)