"""
Anomaly.py - Calculate and plot Z boson mass variation in the 331 model.

This script computes the Z boson mass in the 331 extension of the Standard Model
as a function of the VEV parameter v2, and plots its variation.
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Constants ---
MZ_ESM: float = 91.1876  # GeV
G_COUPLING: float = 0.652
SIN2W: float = 0.23126

def calculate_mz_331(v2: np.ndarray, mz_esm: float, g: float, sin2w: float) -> np.ndarray:
    """
    Calculate MZ in the 331 model for a given array of v2 values.

    Parameters
    ----------
    v2 : np.ndarray
        Array of v2 values [GeV]
    mz_esm : float
        Standard Model Z boson mass [GeV]
    g : float
        Gauge coupling constant
    sin2w : float
        Weinberg angle squared

    Returns
    -------
    np.ndarray
        Array of calculated MZ values [GeV]
    """
    prefactor = (g ** 2 / 4) * (1 / (1 - sin2w)) * 2
    return np.sqrt(mz_esm ** 2 + prefactor * v2 ** 2)

def plot_mz_vs_v2(v2: np.ndarray, mz_331: np.ndarray, mz_esm: float, save_path: str = None):
    """
    Plot MZ vs v2 for the 331 model.

    Parameters
    ----------
    v2 : np.ndarray
        v2 range [GeV]
    mz_331 : np.ndarray
        Calculated MZ values
    mz_esm : float
        Standard Model value
    save_path : str, optional
        If provided, saves the figure to this path instead of showing.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(v2, mz_331, label="331 Model", linewidth=2)
    plt.axhline(mz_esm, color="red", linestyle="--", label="SM Value")
    plt.xlabel(r'$v_2$ [GeV]')
    plt.ylabel(r'$M_Z$ [GeV]')
    plt.title('Variación de $M_Z$ (modelo 331) vs $v_2$')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    else:
        plt.show()

def main():
    # --- v2 range ---
    v2 = np.linspace(0, 50, 500)  # GeV

    # --- Calculate MZ in 331 model ---
    mz_331 = calculate_mz_331(v2, MZ_ESM, G_COUPLING, SIN2W)

    # --- Plot ---
    plot_mz_vs_v2(v2, mz_331, MZ_ESM)

if __name__ == "__main__":
    main()
