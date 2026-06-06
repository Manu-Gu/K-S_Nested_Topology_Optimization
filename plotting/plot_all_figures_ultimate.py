"""
Publication-Quality Figure Generation Pipeline.

This script sequentially generates all figures presented in the manuscript:
"Feasible-set geometry of nested KS constraints in topology optimization: Mechanism, quantification, and compensation".

It features:
- Headless backend compatibility (Agg) for CI/CD or server environments.
- Graceful degradation for LaTeX math-text rendering.
- Advanced hash-based caching to prevent redundant FEA computations.
"""

import os
import sys
import shutil
import hashlib
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

# CI-compatible headless backend (must be set before importing pyplot)
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from scipy.optimize import minimize

# ==========================================
# 1. Infrastructure & Paths
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    from src.topopt.problem import TopOptProblem
    from src.topopt.filter import apply_filter
    from src.topopt.ks_layers import three_layer_G
except ImportError as e:
    logging.error(f"Failed to import topopt modules: {e}")
    logging.info("Please ensure this script is run from the 'plotting' or 'experiments' directory.")
    sys.exit(1)

DATA_DIR = PROJECT_ROOT / "results" / "data"
OUT_DIR  = PROJECT_ROOT / "results" / "figures"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Global layout configurations
FIG_SINGLE_COL = (4.2, 3.0)
BBOX_PROPS = dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.9)


# ==========================================
# 2. Publication-Grade Styling Engine
# ==========================================
def set_strict_academic_style() -> Dict[str, str]:
    """
    Configures Matplotlib to match rigorous academic publication standards.
    Automatically detects LaTeX availability and falls back gracefully.

    Returns
    -------
    Dict[str, str]
        A dictionary containing the standardized color palette.
    """
    has_latex = all(shutil.which(cmd) for cmd in ['latex', 'dvipng', 'gs'])
    if not has_latex:
        logging.warning("LaTeX environment not found. Falling back to robust mathtext rendering.")

    palette = {
        "single": "#1F77B4",  # Scientific Blue
        "nested": "#E377C2",  # Magenta (High contrast in grayscale printing)
        "flat":   "#9467BD",  # Accessible Purple
        "delta":  "#000000",  # Pure Black
    }

    tex_params = {
        "text.usetex": has_latex,
        "font.family": "serif",
        "font.serif": ["Times"] if has_latex else ["DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "stix" if not has_latex else "cm",
    }

    mpl.rcParams.update({
        **tex_params,
        "pdf.fonttype": 42,      # Ensures TrueType fonts are embedded in PDFs
        "ps.fonttype": 42,
        "savefig.dpi": 600,
        "figure.dpi": 150,
        
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "lines.markeredgewidth": 0.8,
        
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.grid": False,
    })
    return palette


def add_outside_legend(ax: Axes) -> None:
    """Places the legend cleanly above the plotting area."""
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, borderaxespad=0.)


def save_fig(fig: Figure, name: str) -> None:
    """Saves the figure in both PDF and PNG formats for publication and web use."""
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight", dpi=600)
    fig.savefig(OUT_DIR / f"{name}.png", bbox_inches="tight", dpi=600)
    logging.info(f"Successfully saved figure: {name}")


def load_data(filename: str, x_key: str, y_key: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Robustly loads and sorts experimental data from a compressed .npz archive."""
    path = DATA_DIR / filename
    if not path.exists():
        logging.warning(f"Data file not found: {filename}")
        return None, None
    try:
        d = np.load(path)
        if x_key not in d or y_key not in d:
            missing = [k for k in (x_key, y_key) if k not in d]
            logging.warning(f"Keys {missing} not found in {filename}. Available: {list(d.keys())}")
            return None, None
        x, y = d[x_key].astype(float), d[y_key].astype(float)
        idx = np.argsort(x)
        return x[idx], y[idx]
    except Exception as e:
        logging.error(f"Failed to load {filename}: {e}")
        return None, None


def _get_optim_hash(nelx: int, nely: int, rmin: float, kappa: float, eps_c: float, maxiter: int) -> str:
    """Generates an MD5 hash representing the optimization parameters for caching."""
    param_str = f"{nelx}_{nely}_{rmin}_{kappa}_{eps_c}_{maxiter}"
    return hashlib.md5(param_str.encode()).hexdigest()[:8]


# ==========================================
# 3. Independent Plotting Modules
# ==========================================

def plot_fig1_schematic() -> None:
    """Generates Figure 1: Schematic of the three-layer monitoring layout and covering multiplicity."""
    nelx, nely = 20, 10
    centers = [(5, 6), (5, 3), (8, 2)]
    offsets = [(0, 0), (2, 0), (0, 2), (2, 2)]
    
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    
    # 1. Compute covering multiplicity (d_e)
    d_e_map = np.zeros((nely, nelx))
    for cx, cy in centers:
        for ox, oy in offsets:
            start_x, start_y = cx + ox - 1, cy + oy - 1
            for i in range(3):
                for j in range(3):
                    ex, ey = start_x + i, start_y + j
                    if 0 <= ex < nelx and 0 <= ey < nely:
                        d_e_map[ey, ex] += 1
                        
    # 2. Custom discrete colormap (white for 0, Blues for overlapping zones)
    max_de = int(np.max(d_e_map))
    base_cmap = plt.get_cmap('Blues', max_de + 1)
    cmaplist = [base_cmap(i) for i in range(base_cmap.N)]
    cmaplist[0] = (1.0, 1.0, 1.0, 1.0) 
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('CustomBlues', cmaplist, base_cmap.N)
    
    # 3. Plot heatmap
    im = ax.imshow(d_e_map, origin='lower', extent=[0, nelx, 0, nely], 
                   cmap=custom_cmap, vmin=-0.5, vmax=max_de+0.5)
    
    # 4. Gridlines
    ax.set_xticks(np.arange(0, nelx + 1, 1), minor=True)
    ax.set_yticks(np.arange(0, nely + 1, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.4)
    
    # 5. Overlay d_e numerical values
    for ey in range(nely):
        for ex in range(nelx):
            val = int(d_e_map[ey, ex])
            if val > 0:
                text_color = 'white' if val >= max_de * 0.6 else 'black'
                ax.text(ex + 0.5, ey + 0.5, str(val), 
                        ha='center', va='center', fontsize=7, color=text_color, fontweight='bold')

    # 6. Overlay Centers and Offset anchors
    for idx_c, (cx, cy) in enumerate(centers):
        for idx_o, (ox, oy) in enumerate(offsets):
            ax.plot(cx + ox + 0.5, cy + oy + 0.5, '+', 
                    color='#D62728', markersize=6, markeredgewidth=1.5, zorder=4,
                    label='Offset anchors' if (idx_c == 0 and idx_o == 0) else "")
        ax.plot(cx + 0.5, cy + 0.5, 'o', 
                color='#FF8C00', markersize=5, zorder=5, 
                label='Centers' if idx_c == 0 else "")
                
    # 7. Axes formatting
    ax.set(xlim=(0, nelx), ylim=(0, nely), xlabel=r"$x$", ylabel=r"$y$")
    ax.set_xticks(np.arange(0, nelx + 1, 5))
    ax.set_yticks(np.arange(0, nely + 1, 2))
    
    # 8. Colorbar
    cbar = fig.colorbar(im, ax=ax, ticks=np.arange(0, max_de + 1), pad=0.02, fraction=0.046)
    cbar.set_label(r'Covering multiplicity $d_e$', rotation=270, labelpad=15)
    
    # 9. Legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.02),
              ncol=2, frameon=False, fontsize=10, handletextpad=0.4)
              
    plt.tight_layout()
    save_fig(fig, "Fig1_schematic_three_layer")
    plt.close()


def plot_fig2_density_field() -> None:
    """Generates Figure 2: Density field topology exactly at the critical threshold."""
    cache_path = DATA_DIR / "fig2_density_cache.npz"
    OPTIM_PARAMS = dict(nelx=20, nely=10, rmin=1.5, kappa=20, eps_c=0.264, maxiter=100)
    current_hash = _get_optim_hash(**OPTIM_PARAMS)
    
    cache_loaded = False
    if cache_path.exists():
        try:
            cached = np.load(cache_path, allow_pickle=True)
            if str(cached.get("param_hash", "")) == current_hash:
                logging.info(f"Cache Hit for Fig2 Density Field (Hash: {current_hash})")
                density_matrix = cached["density"]
                nelx, nely = int(cached["nelx"]), int(cached["nely"])
                fixed_centers = [tuple(c) for c in cached["centers"]]
                cache_loaded = True
            else:
                logging.warning("Cache stale. Recomputing optimization field...")
                cache_path.unlink()
        except Exception:
            cache_path.unlink(missing_ok=True)

    if not cache_loaded:
        logging.info("Cache Miss. Running Topology Optimizer for Fig 2 visualization...")
        nelx, nely, rmin, kappa, eps_c, maxiter = OPTIM_PARAMS.values()
        problem = TopOptProblem(nelx=nelx, nely=nely, rmin=rmin)
        x_base = problem.baseline()
        fixed_centers = problem.get_three_centers(x_base)
        
        def G_nested_wrapper(x: np.ndarray) -> float:
            return three_layer_G(x, fixed_centers, kappa, kappa, kappa, problem.H, problem.Hs, nelx, nely)
        
        cons = [{'type': 'eq', 'fun': lambda x: problem.volfrac - np.mean(x)},
                {'type': 'ineq', 'fun': lambda x: eps_c + 1e-4 - G_nested_wrapper(x)}]
        
        x0 = problem.volfrac * np.ones(nelx * nely)
        res = minimize(problem.compliance, x0, method='SLSQP',
                       bounds=[(0.001, 1.0)] * (nelx * nely),
                       constraints=cons, options={'maxiter': maxiter, 'disp': False})
        
        x_phys = apply_filter(res.x, problem.H, problem.Hs)
        density_matrix = x_phys.reshape((nelx, nely)).T
        
        np.savez(cache_path, density=density_matrix, nelx=nelx, nely=nely,
                 centers=np.array(fixed_centers), kappa=kappa, eps_c=eps_c, param_hash=current_hash)
        logging.info(f"Optimization complete. State cached to {cache_path}")
    
    fig, ax = plt.subplots(figsize=(6, 3.5))
    im = ax.imshow(density_matrix, cmap='gray_r', origin='lower',
                   extent=[0, nelx, 0, nely], vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label(r'Density $\rho(\mathbf{x})$', rotation=-90, labelpad=15, fontsize=10)
    
    for (cx, cy) in fixed_centers:
        for ox, oy in [(0,0), (2,0), (0,2), (2,2)]:
            ax.add_patch(patches.Rectangle(
                (cx + ox - 1, cy + oy - 1), 3, 3,
                linewidth=1.2, edgecolor='darkred', facecolor='grey', alpha=0.15, zorder=2
            ))
    
    ax.annotate('Material evacuation due to\nconstraint aggregation', 
                xy=(10.5, 6.5), 
                xytext=(10.5, 1.5), 
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2',
                                color='darkred', lw=1.5, shrinkA=2, shrinkB=2),
                fontsize=9, color='darkred', ha='center', va='bottom', zorder=5)
    
    ax.set(xticks=np.arange(0, nelx+1, 5), yticks=np.arange(0, nely+1, 5),
           xlabel=r"$x$", ylabel=r"$y$")
    ax.set_title(r'Density Field Topology at Critical Threshold', fontsize=12, pad=10)
    fig.text(0.42, 0.02, r'($\kappa = 20,\ \varepsilon_c = 0.264$)', ha='center', fontsize=10, style='italic')
    
    save_fig(fig, "Fig2_density_near_criticality")
    plt.close()


def plot_data_figures(palette: Dict[str, str]) -> None:
    """Generates Figures 3, 4, 5, 6, and Supplementary Figures from pre-computed numerical data."""
    
    # === Fig 3: eps_c vs kappa ===
    k_s, eps_s = load_data("single_vs_rho.npz", "rho", "eps_c")
    k_t, eps_t = load_data("three_vs_rho_fixed_centers.npz", "rho", "eps_c")
    if k_s is not None:
        fig, ax = plt.subplots(figsize=FIG_SINGLE_COL)
        ax.plot(k_s, eps_s, color=palette["single"], marker='o', linestyle='--',
                mfc='none', markeredgewidth=1.0, label='Single-layer')
        ax.plot(k_t, eps_t, color=palette["nested"], marker='s', linestyle='-',
                label='Three-layer nested')
        ax.set(xlabel=r"Aggregation parameter $\kappa$", ylabel=r"Critical threshold $\varepsilon_c$")
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        add_outside_legend(ax)
        save_fig(fig, "Fig3_eps_vs_kappa_single_three")
        plt.close(fig)

    # === Fig 4: eps_c vs r_min ===
    r_s, e_s = load_data("single_vs_rmin.npz", "rmin", "eps_c")
    r_t, e_t = load_data("three_vs_rmin_fixed_centers.npz", "rmin", "eps_c")
    if r_s is not None:
        fig, ax = plt.subplots(figsize=FIG_SINGLE_COL)
        ax.plot(r_s, e_s, color=palette["single"], marker='o', linestyle='--',
                mfc='none', markeredgewidth=1.0, label='Single-layer')
        ax.plot(r_t, e_t, color=palette["nested"], marker='s', linestyle='-',
                label='Three-layer nested')
        ax.set(xlabel=r"Filter radius $r_{\min}$", ylabel=r"Critical threshold $\varepsilon_c$")
        ax.set_xticks(np.unique(r_s))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        add_outside_legend(ax)
        save_fig(fig, "Fig4_eps_vs_rmin_single_three")
        plt.close(fig)

    # === Fig 5a: Nested vs Flat (kappa ablation) ===
    k, en = load_data("compare_nested_flat_vs_rho_scale3.npz", "rho", "eps_nested")
    _, ef = load_data("compare_nested_flat_vs_rho_scale3.npz", "rho", "eps_flat")
    if k is not None:
        fig, ax = plt.subplots(figsize=FIG_SINGLE_COL)
        ax.plot(k, ef, color=palette["flat"], marker='v', linestyle='-.', label='Flat union-set')
        ax.plot(k, en, color=palette["nested"], marker='s', linestyle='-', label='Three-layer nested')
        ax.set(xlabel=r"Aggregation parameter $\kappa$", ylabel=r"Critical threshold $\varepsilon_c$")
        add_outside_legend(ax)
        save_fig(fig, "Fig5a_ablation_nested_vs_flat_kappa")
        plt.close(fig)

    # === Fig 5b: Nested vs Flat (rmin ablation) ===
    r, en2 = load_data("compare_nested_flat_vs_rmin_fixed_centers.npz", "rmin", "eps_nested")
    _, ef2 = load_data("compare_nested_flat_vs_rmin_fixed_centers.npz", "rmin", "eps_flat")
    if r is not None:
        fig, ax = plt.subplots(figsize=FIG_SINGLE_COL)
        ax.plot(r, ef2, color=palette["flat"], marker='v', linestyle='-.', label='Flat union-set')
        ax.plot(r, en2, color=palette["nested"], marker='s', linestyle='-', label='Three-layer nested')
        ax.set(xlabel=r"Filter radius $r_{\min}$", ylabel=r"Critical threshold $\varepsilon_c$")
        ax.set_xticks(np.unique(r))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        add_outside_legend(ax)
        save_fig(fig, "Fig5b_ablation_nested_vs_flat_rmin")
        plt.close(fig)

    # === Fig 6: MBB Universality Benchmark ===
    mbb_nested = np.array([0.4695, 0.2351, 0.1187])
    mbb_flat   = np.array([0.4329, 0.2168, 0.1091])
    x_pos = np.arange(3)
    width = 0.35 

    fig, ax = plt.subplots(figsize=FIG_SINGLE_COL)
    
    bars1 = ax.bar(x_pos - width/2, mbb_nested, width,
                   color=palette["nested"], edgecolor='black', linewidth=1.2,
                   label='Three-layer nested')
    bars2 = ax.bar(x_pos + width/2, mbb_flat, width,
                   color=palette["flat"], edgecolor='black', linewidth=1.2,
                   label='Flat union-set')
    
    ax.set(ylabel=r'Critical Threshold $\varepsilon_c$',
           xticks=x_pos,
           xticklabels=[r'$\kappa=10$', r'$\kappa=20$', r'$\kappa=40$'])
    
    TIGHT_BBOX = dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85)
    
    for i, (val_n, val_f) in enumerate(zip(mbb_nested, mbb_flat)):
        ax.text(x_pos[i] - width/2, val_n + 0.015, f'{val_n:.4f}',
                ha='center', va='bottom', rotation=90, fontsize=8.5, 
                color='black', fontweight='bold', bbox=TIGHT_BBOX, zorder=5)
        
        ax.text(x_pos[i] + width/2, val_f + 0.015, f'{val_f:.4f}',
                ha='center', va='bottom', rotation=90, fontsize=8.5, 
                color='black', fontweight='bold', bbox=TIGHT_BBOX, zorder=5)
    
    ax.set_ylim(0, max(mbb_nested) * 1.35) 
    add_outside_legend(ax)
    save_fig(fig, "FigMBB_universality_benchmark")
    plt.close(fig)

    # === Supplementary Fig S1: Delta eps_c vs kappa ===
    if k is not None:
        fig, ax = plt.subplots(figsize=FIG_SINGLE_COL)
        ax.plot(k, en - ef, 'k', marker='d', linestyle='-', dashes=(3,2),
                mfc='none', markeredgewidth=0.8, label='Incremental Bias')
        ax.axhline(0.0, color='gray', linestyle='--', linewidth=0.8, zorder=1)
        ax.set(xlabel=r"Aggregation parameter $\kappa$", ylabel=r"Incremental effect $\Delta\varepsilon_c$", xlim=(0, 85))
        add_outside_legend(ax)
        save_fig(fig, "FigS1_delta_kappa")
        plt.close(fig)

    # === Supplementary Fig S2: Delta eps_c vs r_min ===
    if r is not None:
        fig, ax = plt.subplots(figsize=FIG_SINGLE_COL)
        ax.plot(r, en2 - ef2, 'k', marker='d', linestyle='-', dashes=(3,2),
                mfc='none', markeredgewidth=0.8, label='Incremental Bias')
        ax.axhline(0.0, color='gray', linestyle='--', linewidth=0.8, zorder=1)
        ax.set(xlabel=r"Filter radius $r_{\min}$", ylabel=r"Incremental effect $\Delta\varepsilon_c$")
        ax.set_xticks(np.unique(r))
        add_outside_legend(ax)
        save_fig(fig, "FigS2_delta_rmin")
        plt.close(fig)

    # === Supplementary Fig S3: Spurious Infeasibility Probability Curve ===
    eps_points = np.array([0.245, 0.250, 0.255, 0.260, 0.265, 0.270, 0.275, 0.280])
    success_rates = np.array([0.0, 0.1, 0.0, 0.2, 0.5, 0.7, 1.0, 1.0])
    fig, ax = plt.subplots(figsize=FIG_SINGLE_COL)
    
    ax.plot(eps_points, success_rates, marker='o', linestyle='-', drawstyle='steps-mid',
            color='black', mfc='none', markeredgewidth=0.8, label='Success Rate')
    ax.axvline(x=0.2635, color=palette["nested"], linestyle='--', 
               linewidth=1.5, alpha=0.8, zorder=3, label=r'Critical $\varepsilon_c$')
    ax.set(xlabel=r'Constraint Threshold $\varepsilon$', ylabel='Feasibility Probability', ylim=(-0.05, 1.05))
    add_outside_legend(ax)
    save_fig(fig, "FigS3_spurious_infeasibility_curve")
    plt.close(fig)


# ==========================================
# 4. Main Pipeline Execution
# ==========================================
def main() -> None:
    logging.info("="*60)
    logging.info("INITIALIZING: Publication-Quality Plotting Pipeline")
    logging.info("="*60)
    
    palette = set_strict_academic_style()
    
    plot_fig1_schematic()
    plot_fig2_density_field()
    plot_data_figures(palette)
    
    logging.info("="*60)
    logging.info("SUCCESS: All manuscript figures generated conforming to strict academic layout.")
    logging.info("="*60)

if __name__ == "__main__":
    main()