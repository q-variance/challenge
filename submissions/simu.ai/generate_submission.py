"""
Generate submission for Q-Variance Challenge

This script:
1. Simulates price data using the regime mixture Q-variance model
2. Processes the CSV through data_loader_csv.py to generate dataset.parquet
3. Scores the submission using score_submission.py
"""
import os
os.environ["MPLBACKEND"] = "Agg"  


import sys
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path

# Set matplotlib to non-interactive backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode

from scipy.optimize import curve_fit
from scipy.stats import norm, poisson
from sklearn.metrics import r2_score

# Add parent directories to path to import challenge code
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'code'))

from model_simulation import generate_price_csv

# Configuration
SUBMISSION_DIR = Path(__file__).parent
CHALLENGE_ROOT = Path(__file__).parent.parent.parent
DATA_LOADER_SCRIPT = CHALLENGE_ROOT / 'code' / 'data_loader_csv.py'
SCORE_SCRIPT = CHALLENGE_ROOT / 'code' / 'score_submission.py'

# Model parameters for regime mixture Q-variance model
# Based on the model structure: σ²(z) = σ₀² + (z - z₀)²/2
# Parameters from model2.ipynb:
SIGMA0 = 0.282388     # Baseline variance scale (σ₀)
MU = 0.023182         # Drift per year (z₀)
N_DAYS = 5_000_000    # Number of trading days to simulate
SAMPLES_PER_DAY = 4   # Internal steps per day for simulation granularity
MAX_WINDOW_DAYS = 130 # Maximum window size in days (for regime length heuristic)

# Note: The model uses regime-switching variance with Gamma-distributed precision
# Regime lengths are geometric with mean ≈ 10 * max_window_days


def qvar(z, s0, zoff):
    """Q-variance function: σ²(z) = σ₀² + (z - z₀)²/2"""
    return (s0**2 + (z - zoff)**2 / 2)


def quantum_density(z, sig0, zoff=0.0):
    """Quantum density function — returns plain array for curve_fit"""
    ns = np.arange(0, 6)
    qdn = np.zeros_like(z, dtype=float)
    sigvec = sig0 * np.sqrt(2 * ns + 1)
    means = zoff * np.ones_like(ns) - sigvec**2/2  # no drift term in pure Q-Variance

    for n in ns:
        weight = poisson.pmf(n, mu=0.5)
        qdn += weight * norm.pdf(z, loc=means[n], scale=sigvec[n])
    return qdn


def generate_figures(dataset_path, output_dir):
    """Generate Figure_1.png and Figure_5.png for the submission"""
    print(f"Loading dataset from {dataset_path}...")
    data = pd.read_parquet(dataset_path)
    data["var"] = data.sigma**2
    
    print(f"Loaded {len(data)} windows")
    
    # ===== Figure 1: Q-Variance scatter plot =====
    print("Generating Figure_1.png...")
    
    zmax = 0.6
    delz = 0.025 * 2
    nbins = int(2 * zmax / delz + 1)
    bins = np.linspace(-zmax, zmax, nbins)
    
    # Create binned data
    binned = (data.assign(z_bin=pd.cut(data.z, bins=bins, include_lowest=True))
                   .groupby('z_bin', observed=False)
                   .agg(z_mid=('z', 'mean'), var=('var', 'mean'))
                   .dropna())
    
    # Fit q-variance curve
    popt, _ = curve_fit(qvar, binned.z_mid, binned["var"], p0=[0.25, 0.02])
    fitted = qvar(binned.z_mid, popt[0], popt[1])
    r2 = 1 - np.sum((binned["var"] - fitted)**2) / np.sum((binned["var"] - binned["var"].mean())**2)
    
    print(f"  σ₀ = {popt[0]:.4f}  zoff = {popt[1]:.4f}  R² = {r2:.4f}")
    
    # Plot
    markfac = 1
    plt.figure(figsize=(9, 7))
    plt.scatter(data.z, data['var'], c='steelblue', alpha=markfac*0.1, s=markfac*1, edgecolor='none')
    plt.plot(binned.z_mid, binned['var'], 'b-', lw=3, label='Binned data')
    plt.plot(binned.z_mid, fitted, 'red', lw=3, 
             label=f'σ₀ = {popt[0]:.3f}, zoff = {popt[1]:.3f}, R² = {r2:.3f}')
    
    plt.xlabel('z (scaled log return)', fontsize=12)
    plt.ylabel('Annualised variance', fontsize=12)
    plt.title('Q-Variance: all data T=1 to 26 weeks', fontsize=14)
    plt.xlim(-zmax, zmax)
    plt.ylim(0.0, 0.35)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    figure1_path = output_dir / 'Figure_1.png'
    plt.savefig(figure1_path, dpi=300, bbox_inches='tight')
    plt.close('all')  # Close all figures
    print(f"  Saved to {figure1_path}")
    
    # ===== Figure 5: Time-invariant distribution =====
    print("Generating Figure_5.png...")
    
    zlim = 2
    zbins = np.linspace(-zlim, zlim, 51)
    zmid = (zbins[:-1] + zbins[1:]) / 2
    
    # Histogram for all data
    counts, _ = np.histogram(data["z"], bins=zbins, density=True)
    
    # Fit quantum model
    p0 = [0.62, 0.0]
    popt_q, _ = curve_fit(quantum_density, zmid, counts, p0=p0, bounds=(0, [2.0, 0.5]))
    sig0_fit, zoff_fit = popt_q
    
    # Predict on fine grid
    z_fine = np.linspace(-zlim, zlim, 1000)
    q_pred_fine = quantum_density(z_fine, *popt_q)
    
    # Predict on histogram bin centers for R²
    q_pred_hist = quantum_density(zmid, *popt_q)
    r2_all = r2_score(counts, q_pred_hist)
    
    print(f"  Fit: σ₀ = {sig0_fit:.4f}, zoff = {zoff_fit:.4f}, R² = {r2_all:.4f}")
    
    # Plot with different periods
    TVEC = [5, 10, 20, 40, 80]
    
    plt.figure(figsize=(9, 7))
    plt.plot(z_fine, q_pred_fine,
             color='red', lw=4,
             label=f'Q-Variance fit: σ₀ = {sig0_fit:.3f}, R² = {r2_all:.4f}')
    
    for Tcur in TVEC:
        datacur = data[(data["T"] == Tcur)].copy()
        if len(datacur) > 0:
            counts_cur, _, _ = plt.hist(datacur["z"], bins=zbins, density=True, visible=False)
            r2_cur = r2_score(counts_cur, q_pred_hist)  # use fit for whole data set
            colcur = str(Tcur / (max(TVEC) + 20))
            plt.plot(zmid, counts_cur, c=colcur, lw=2, 
                    label=f'T = {Tcur/5:.0f}, R² = {r2_cur:.3f}')
    
    plt.title('Q-Variance: T dependence', fontsize=18, pad=20)
    plt.xlabel('Scaled log-return z', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.xlim(-1.2, 1.2)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    figure5_path = output_dir / 'Figure_5.png'
    plt.savefig(figure5_path, dpi=300, bbox_inches='tight')
    plt.close('all')  # Close all figures
    print(f"  Saved to {figure5_path}")


def run_data_loader():
    """Run data_loader_csv.py to process the CSV and generate dataset.parquet"""
    print("\n" + "="*60)
    print("Step 2: Processing CSV through data_loader_csv.py")
    print("="*60)
    
    # Change to challenge root directory
    os.chdir(CHALLENGE_ROOT)
    
    # The data_loader_csv.py expects variance_timeseries.csv in the current directory
    # and will create dataset.parquet
    result = subprocess.run(
        [sys.executable, str(DATA_LOADER_SCRIPT)],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    # Move dataset.parquet to submission directory
    dataset_file = CHALLENGE_ROOT / 'dataset.parquet'
    if dataset_file.exists():
        target = SUBMISSION_DIR / 'dataset.parquet'
        dataset_file.rename(target)
        print(f"\nMoved dataset.parquet to {target}")
    else:
        print("Warning: dataset.parquet was not created")


def main():
    """Main execution function"""
    print("="*60)
    print("Q-Variance Challenge Submission Generator")
    print("Team: 2001: A State-Space Odyssey")
    print("="*60)
    
    # Step 1: Generate price simulation
    print("\n" + "="*60)
    print("Step 1: Generating price simulation")
    print("="*60)
    print(f"Parameters: σ₀ = {SIGMA0:.4f}, μ = {MU:.4f}")
    print(f"Simulating {N_DAYS:,} days (~{N_DAYS/252:.1f} years)")
    print(f"Samples per day: {SAMPLES_PER_DAY}, Max window: {MAX_WINDOW_DAYS} days")
    
    # Generate CSV in challenge root (where data_loader_csv.py expects it)
    csv_file = CHALLENGE_ROOT / 'variance_timeseries.csv'
    generate_price_csv(
        sigma0=SIGMA0,
        mu=MU,
        n_days=N_DAYS,
        samples_per_day=SAMPLES_PER_DAY,
        max_window_days=MAX_WINDOW_DAYS,
        output_file=str(csv_file),
        seed=42  # For reproducibility
    )
    
    # Step 2: Process through data_loader_csv.py
    run_data_loader()
    
    # Step 3: Score the submission
    print("\n" + "="*60)
    print("Step 3: Scoring submission")
    print("="*60)
    
    # Modify score_submission.py temporarily to read from submission directory
    # Or copy dataset.parquet to challenge root for scoring
    dataset_file = SUBMISSION_DIR / 'dataset.parquet'
    if dataset_file.exists():
        # Copy to challenge root for scoring
        import shutil
        temp_dataset = CHALLENGE_ROOT / 'dataset.parquet'
        shutil.copy(dataset_file, temp_dataset)
        
        # Run scoring
        os.chdir(CHALLENGE_ROOT)
        result = subprocess.run(
            [sys.executable, str(SCORE_SCRIPT)],
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        
        # Clean up temp file
        if temp_dataset.exists():
            temp_dataset.unlink()
    
    # Step 4: Generate figures
    print("\n" + "="*60)
    print("Step 4: Generating figures")
    print("="*60)
    generate_figures(SUBMISSION_DIR / 'dataset.parquet', SUBMISSION_DIR)
    
    print("\n" + "="*60)
    print("Submission generation complete!")
    print("="*60)
    print(f"Submission directory: {SUBMISSION_DIR}")
    print(f"Files created:")
    print(f"  - {SUBMISSION_DIR / 'dataset.parquet'}")
    print(f"  - {SUBMISSION_DIR / 'README.md'}")
    print(f"  - {SUBMISSION_DIR / 'Figure_1.png'}")
    print(f"  - {SUBMISSION_DIR / 'Figure_5.png'}")


if __name__ == '__main__':
    main()

