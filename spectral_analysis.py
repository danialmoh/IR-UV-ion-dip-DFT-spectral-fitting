"""
Quantitative spectral comparison pipeline:
  1. Cosine similarity ranking
  2. NNLS linear combination fitting
  3. Sequential addition (explained variance)
  4. BIC model selection
  5. Residual analysis + summary plot

Compares experimental m/z 140 IRMPD spectrum against 
B3LYP/def2-TZVP scaled DFT spectra (41 isomers).
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from scipy.optimize import nnls
from itertools import combinations

# === PATHS ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_FILE = os.path.join(BASE_DIR, "mass_channel_IR_spectra_wn401-1576_2ch-negativenotcropped.csv")
DFT_DIR = os.path.join(BASE_DIR, "b3lyp_def2-tzvp", "scaled")
OUTPUT_DIR = os.path.join(BASE_DIR, "b3lyp_def2-tzvp", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === SETTINGS ===
WAVENUMBER_RANGE = (400, 1600)  # cm-1

# === Load experimental spectrum ===
exp_df = pd.read_csv(EXP_FILE)
exp_wn = exp_df["Wavenumber"].values
exp_intensity = exp_df["m/z 140.0 -ln(depl)"].values

# Trim to range
mask = (exp_wn >= WAVENUMBER_RANGE[0]) & (exp_wn <= WAVENUMBER_RANGE[1])
exp_wn = exp_wn[mask]
exp_intensity = exp_intensity[mask]

# Normalize to [0, 1]
exp_max = exp_intensity.max()
exp_norm = exp_intensity / exp_max if exp_max > 0 else exp_intensity

# === Collect DFT spectra ===
convoluted_files = sorted(glob.glob(os.path.join(DFT_DIR, "CID_*_scaled_0.95.csv")))

molecules = []
dft_matrix = []  # each row = one DFT spectrum interpolated onto exp grid

for conv_file in convoluted_files:
    basename = os.path.basename(conv_file)
    cid_match = re.match(r"CID_(\d+)_(.+)_scaled_0\.95\.csv", basename)
    if not cid_match:
        continue
    cid = cid_match.group(1)
    name = cid_match.group(2).replace("_", " ")

    dft_df = pd.read_csv(conv_file, comment="#", header=None,
                         names=["Wavenumber", "Fundamentals", "Overtones",
                                "Combinations", "Total"])
    dft_wn = dft_df["Wavenumber"].values
    dft_total = dft_df["Total"].values

    # Interpolate onto experimental wavenumber grid
    dft_interp = np.interp(exp_wn, dft_wn, dft_total)

    molecules.append({"cid": cid, "name": name, "file": conv_file})
    dft_matrix.append(dft_interp)

dft_matrix = np.array(dft_matrix)  # shape: (n_molecules, n_wavenumbers)
n_mol = len(molecules)
print(f"Loaded {n_mol} DFT spectra.")

# ============================================================
# STEP 1: Cosine Similarity Ranking
# ============================================================
print("\n" + "="*60)
print("STEP 1: Cosine Similarity Ranking")
print("="*60)

cosine_scores = []
for i in range(n_mol):
    dot = np.dot(exp_norm, dft_matrix[i])
    norm_exp = np.linalg.norm(exp_norm)
    norm_dft = np.linalg.norm(dft_matrix[i])
    if norm_exp > 0 and norm_dft > 0:
        cos_sim = dot / (norm_exp * norm_dft)
    else:
        cos_sim = 0.0
    cosine_scores.append(cos_sim)

cosine_scores = np.array(cosine_scores)
ranking = np.argsort(cosine_scores)[::-1]  # descending

print(f"\n{'Rank':<5} {'CID':<12} {'Name':<45} {'Cosine Sim':>10}")
print("-" * 75)
for r, idx in enumerate(ranking[:15]):
    print(f"{r+1:<5} {molecules[idx]['cid']:<12} {molecules[idx]['name'][:44]:<45} {cosine_scores[idx]:>10.4f}")

# ============================================================
# STEP 2: NNLS Linear Combination Fitting (all 41)
# ============================================================
print("\n" + "="*60)
print("STEP 2: NNLS Linear Combination Fitting")
print("="*60)

# NNLS: minimize ||exp - A @ x||^2, x >= 0
# A = DFT matrix transposed (columns = DFT spectra)
A = dft_matrix.T  # shape: (n_wavenumbers, n_molecules)
coeffs_nnls, residual_nnls = nnls(A, exp_norm)

# Normalize coefficients to sum to 1
coeff_sum = coeffs_nnls.sum()
if coeff_sum > 0:
    coeffs_normalized = coeffs_nnls / coeff_sum
else:
    coeffs_normalized = coeffs_nnls

# Sort by contribution
nnls_ranking = np.argsort(coeffs_normalized)[::-1]

print(f"\nNNLS Residual norm: {residual_nnls:.4f}")
print(f"Sum of raw coefficients: {coeff_sum:.4f}")
print(f"\n{'Rank':<5} {'CID':<12} {'Name':<45} {'Weight':>8} {'Raw Coeff':>10}")
print("-" * 85)
non_zero_count = 0
for r, idx in enumerate(nnls_ranking):
    if coeffs_normalized[idx] < 1e-4:
        break
    non_zero_count += 1
    print(f"{r+1:<5} {molecules[idx]['cid']:<12} {molecules[idx]['name'][:44]:<45} "
          f"{coeffs_normalized[idx]:>7.1%} {coeffs_nnls[idx]:>10.4f}")

print(f"\nNon-zero contributors: {non_zero_count} / {n_mol}")

# Reconstructed spectrum
reconstructed_nnls = A @ coeffs_nnls

# R² score
ss_res = np.sum((exp_norm - reconstructed_nnls)**2)
ss_tot = np.sum((exp_norm - exp_norm.mean())**2)
r2_full = 1 - ss_res / ss_tot
print(f"R² (full NNLS, all components): {r2_full:.4f}")

# ============================================================
# STEP 3: Sequential Addition — Explained Variance
# ============================================================
print("\n" + "="*60)
print("STEP 3: Sequential Addition (Cumulative Explained Variance)")
print("="*60)

cumulative_r2 = []
cumulative_labels = []
added_indices = []

for r, idx in enumerate(nnls_ranking):
    if coeffs_normalized[idx] < 1e-4:
        break
    added_indices.append(idx)
    # Re-fit NNLS with only the top-r components
    A_sub = dft_matrix[added_indices].T
    coeffs_sub, _ = nnls(A_sub, exp_norm)
    recon_sub = A_sub @ coeffs_sub
    ss_res_sub = np.sum((exp_norm - recon_sub)**2)
    r2_sub = 1 - ss_res_sub / ss_tot
    cumulative_r2.append(r2_sub)
    cumulative_labels.append(f"CID {molecules[idx]['cid']}")
    print(f"  + {molecules[idx]['cid']:<12} {molecules[idx]['name'][:35]:<35}  "
          f"Cumulative R² = {r2_sub:.4f}  (ΔR² = {r2_sub - (cumulative_r2[-2] if len(cumulative_r2) > 1 else 0):.4f})")

# ============================================================
# STEP 4: BIC Model Selection
# ============================================================
print("\n" + "="*60)
print("STEP 4: BIC Model Selection")
print("="*60)

n_points = len(exp_norm)
bic_values = []

for k in range(1, min(non_zero_count + 1, 16)):  # test up to 15 components
    top_k_indices = [nnls_ranking[i] for i in range(k)]
    A_k = dft_matrix[top_k_indices].T
    coeffs_k, _ = nnls(A_k, exp_norm)
    recon_k = A_k @ coeffs_k
    rss = np.sum((exp_norm - recon_k)**2)
    # BIC = n*ln(RSS/n) + k*ln(n)
    bic = n_points * np.log(rss / n_points) + k * np.log(n_points)
    bic_values.append(bic)
    print(f"  k={k:2d}  BIC = {bic:8.2f}  RSS = {rss:.4f}")

best_k = np.argmin(bic_values) + 1
print(f"\n** Optimal number of components (lowest BIC): k = {best_k} **")

# ============================================================
# STEP 5: Final fit with optimal k + Residual
# ============================================================
print("\n" + "="*60)
print(f"STEP 5: Final Fit with k={best_k} Components")
print("="*60)

best_indices = [nnls_ranking[i] for i in range(best_k)]
A_best = dft_matrix[best_indices].T
coeffs_best, _ = nnls(A_best, exp_norm)
recon_best = A_best @ coeffs_best
residual_best = exp_norm - recon_best

coeffs_best_norm = coeffs_best / coeffs_best.sum() if coeffs_best.sum() > 0 else coeffs_best

ss_res_best = np.sum(residual_best**2)
r2_best = 1 - ss_res_best / ss_tot

print(f"\nFinal R² = {r2_best:.4f}")
print(f"\n{'Component':<5} {'CID':<12} {'Name':<40} {'Population':>10}")
print("-" * 70)
for i, idx in enumerate(best_indices):
    print(f"{i+1:<5} {molecules[idx]['cid']:<12} {molecules[idx]['name'][:39]:<40} {coeffs_best_norm[i]:>9.1%}")

# ============================================================
# OUTPUT: Summary CSV
# ============================================================
summary_data = []
for i in range(n_mol):
    summary_data.append({
        "CID": molecules[i]["cid"],
        "Name": molecules[i]["name"],
        "Cosine_Similarity": cosine_scores[i],
        "NNLS_Coefficient": coeffs_nnls[i],
        "NNLS_Weight_Percent": coeffs_normalized[i] * 100,
        "Cosine_Rank": int(np.where(ranking == i)[0][0]) + 1,
        "NNLS_Rank": int(np.where(nnls_ranking == i)[0][0]) + 1,
    })

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values("NNLS_Weight_Percent", ascending=False).reset_index(drop=True)
summary_csv = os.path.join(OUTPUT_DIR, "140_spectral_analysis_summary.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"\nSummary CSV saved to: {summary_csv}")

# ============================================================
# OUTPUT: Summary Plot (PDF)
# ============================================================
output_pdf = os.path.join(OUTPUT_DIR, "140_spectral_analysis.pdf")

with PdfPages(output_pdf) as pdf:

    # --- Page 1: Cosine similarity bar chart ---
    fig, ax = plt.subplots(figsize=(12, 6))
    top_n = min(20, n_mol)
    top_idx = ranking[:top_n]
    bars = ax.barh(range(top_n), cosine_scores[top_idx], color="steelblue", edgecolor="none")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([f"CID {molecules[i]['cid']} - {molecules[i]['name'][:30]}" for i in top_idx], fontsize=7)
    ax.set_xlabel("Cosine Similarity")
    ax.set_title("Top 20 Isomers by Cosine Similarity to Exp m/z 140", fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    pdf.savefig(fig, dpi=100)
    plt.close(fig)

    # --- Page 2: NNLS weights bar chart ---
    fig, ax = plt.subplots(figsize=(12, 6))
    top_nnls = [nnls_ranking[i] for i in range(min(15, non_zero_count))]
    weights = [coeffs_normalized[i] * 100 for i in top_nnls]
    ax.barh(range(len(top_nnls)), weights, color="darkorange", edgecolor="none")
    ax.set_yticks(range(len(top_nnls)))
    ax.set_yticklabels([f"CID {molecules[i]['cid']} - {molecules[i]['name'][:30]}" for i in top_nnls], fontsize=7)
    ax.set_xlabel("NNLS Weight (%)")
    ax.set_title("Top Contributors by NNLS Linear Combination Fitting", fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    pdf.savefig(fig, dpi=100)
    plt.close(fig)

    # --- Page 3: Sequential R² + BIC ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(range(1, len(cumulative_r2) + 1), cumulative_r2, "o-", color="darkgreen", lw=2)
    ax1.axhline(y=r2_full, color="gray", ls="--", lw=0.8, label=f"Full model R²={r2_full:.3f}")
    ax1.set_xlabel("Number of Components")
    ax1.set_ylabel("Cumulative R²")
    ax1.set_title("Sequential Addition: Explained Variance")
    ax1.legend()
    ax1.set_ylim(0, 1.05)

    ax2.plot(range(1, len(bic_values) + 1), bic_values, "s-", color="crimson", lw=2)
    ax2.axvline(x=best_k, color="gray", ls="--", lw=0.8, label=f"Optimal k={best_k}")
    ax2.set_xlabel("Number of Components (k)")
    ax2.set_ylabel("BIC")
    ax2.set_title("BIC Model Selection")
    ax2.legend()

    plt.tight_layout()
    pdf.savefig(fig, dpi=100)
    plt.close(fig)

    # --- Page 4: Best fit overlay + residual ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(exp_wn, exp_norm, "k-", lw=1.2, label="Experimental m/z 140")
    ax1.plot(exp_wn, recon_best, "r-", lw=1.0, alpha=0.85,
             label=f"NNLS fit (k={best_k}, R²={r2_best:.3f})")

    # Plot individual contributions
    colors = plt.colormaps["tab10"]
    for i, idx in enumerate(best_indices):
        contribution = dft_matrix[idx] * coeffs_best[i]
        ax1.fill_between(exp_wn, 0, contribution, alpha=0.2, color=colors(i),
                         label=f"CID {molecules[idx]['cid']} ({coeffs_best_norm[i]:.0%})")

    ax1.set_xlim(WAVENUMBER_RANGE)
    ax1.set_ylabel("Intensity (norm.)")
    ax1.set_title(f"Best Fit: {best_k}-Component NNLS Decomposition", fontweight="bold")
    ax1.legend(fontsize=7, loc="upper right")

    ax2.plot(exp_wn, residual_best, "k-", lw=0.8)
    ax2.axhline(0, color="gray", ls="--", lw=0.5)
    ax2.fill_between(exp_wn, 0, residual_best, alpha=0.3, color="gray")
    ax2.set_xlim(WAVENUMBER_RANGE)
    ax2.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax2.set_ylabel("Residual")
    ax2.set_title("Residual (Exp - Fit)")

    plt.tight_layout()
    pdf.savefig(fig, dpi=100)
    plt.close(fig)

print(f"Analysis PDF saved to: {output_pdf}")
print("\nDone!")
