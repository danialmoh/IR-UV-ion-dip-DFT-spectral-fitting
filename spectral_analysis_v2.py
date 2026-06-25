"""
Spectral Analysis Pipeline v2 — Methodologically Revised
=========================================================
Compares experimental m/z 140 IRMPD spectrum against
B3LYP/def2-TZVP scaled DFT spectra (41 isomers).

Improvements over v1:
  - Reports "spectral weights", not populations (IRMPD nonlinearity caveat)
  - Collinearity diagnostics (condition number, DFT-DFT Gram matrix)
  - Pearson correlation on smoothed first-derivative spectra for ranking
  - Forward stepwise selection (greedy RSS-optimal) instead of weight-ordered
  - Explicit baseline term in fit via lsq_linear with mixed bounds
  - BIC with effective n (bandwidth-corrected for FELIX resolution)
  - Bootstrap confidence intervals on spectral weights
  - Sensitivity analysis across scaling factors {0.94, 0.95, 0.96, 0.97}

References:
  - NNLS: Barnes et al. (2020) Int. J. Mass Spectrom. 447, 116235
  - Mixture ID: Martynko et al. (2025) arXiv:2602.21308
  - BIC: Schwarz (1978) Ann. Statist. 6(2), 461–464
  - Cosine scoring: Stein & Scott (1994) JASMS 5, 859–866
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
from scipy.optimize import nnls, lsq_linear
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# === PATHS ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_FILE = os.path.join(BASE_DIR, "mass_channel_IR_spectra_wn401-1576_2ch-negativenotcropped.csv")
DFT_DIR = os.path.join(BASE_DIR, "b3lyp_def2-tzvp", "scaled")
OUTPUT_DIR = os.path.join(BASE_DIR, "b3lyp_def2-tzvp", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === SETTINGS ===
WAVENUMBER_RANGE = (400, 1600)  # cm-1
FELIX_FWHM = 10.0  # cm-1, approximate FELIX bandwidth in fingerprint
SCALING_FACTORS = [0.94, 0.95, 0.96, 0.97]  # sensitivity test
DEFAULT_SCALE = 0.95
N_BOOTSTRAP = 500
MAX_K_EXHAUSTIVE = 5  # exhaustive subset search up to this k
MAX_K_FORWARD = 12  # forward stepwise beyond exhaustive

# Effective number of independent spectral points (bandwidth-corrected)
wn_range_span = WAVENUMBER_RANGE[1] - WAVENUMBER_RANGE[0]
N_EFF = int(wn_range_span / FELIX_FWHM)  # ~120 for 1200/10


def load_experimental():
    """Load and pre-process experimental spectrum."""
    exp_df = pd.read_csv(EXP_FILE)
    exp_wn = exp_df["Wavenumber"].values
    exp_intensity = exp_df["m/z 140.0 -ln(depl)"].values
    mask = (exp_wn >= WAVENUMBER_RANGE[0]) & (exp_wn <= WAVENUMBER_RANGE[1])
    exp_wn = exp_wn[mask]
    exp_intensity = exp_intensity[mask]
    exp_max = exp_intensity.max()
    exp_norm = exp_intensity / exp_max if exp_max > 0 else exp_intensity
    return exp_wn, exp_norm


def load_dft_spectra(exp_wn, scale_factor=0.95):
    """Load all DFT spectra, interpolate onto exp grid."""
    pattern = f"CID_*_scaled_{scale_factor:.2f}.csv"
    convoluted_files = sorted(glob.glob(os.path.join(DFT_DIR, pattern)))

    molecules = []
    dft_matrix = []

    for conv_file in convoluted_files:
        basename = os.path.basename(conv_file)
        cid_match = re.match(rf"CID_(\d+)_(.+)_scaled_{scale_factor:.2f}\.csv", basename)
        if not cid_match:
            continue
        cid = cid_match.group(1)
        name = cid_match.group(2).replace("_", " ")

        dft_df = pd.read_csv(conv_file, comment="#", header=None,
                             names=["Wavenumber", "Fundamentals", "Overtones",
                                    "Combinations", "Total"])
        dft_wn = dft_df["Wavenumber"].values
        dft_total = dft_df["Total"].values
        dft_interp = np.interp(exp_wn, dft_wn, dft_total)

        molecules.append({"cid": cid, "name": name, "file": conv_file})
        dft_matrix.append(dft_interp)

    return molecules, np.array(dft_matrix)


def build_design_matrix_with_baseline(dft_matrix, exp_wn, poly_order=1):
    """Build design matrix A with DFT columns + polynomial baseline columns.
    Baseline columns are NOT non-negativity constrained."""
    n_wn = len(exp_wn)
    # Normalize wavenumber to [-1, 1] for numerical stability
    wn_norm = 2 * (exp_wn - exp_wn.min()) / (exp_wn.max() - exp_wn.min()) - 1
    baseline_cols = []
    for p in range(poly_order + 1):
        baseline_cols.append(wn_norm ** p)
    baseline_matrix = np.array(baseline_cols).T  # (n_wn, poly_order+1)
    # A = [DFT columns | baseline columns]
    A = np.hstack([dft_matrix.T, baseline_matrix])
    n_dft = dft_matrix.shape[0]
    n_baseline = baseline_matrix.shape[1]
    return A, n_dft, n_baseline


def fit_with_baseline(A, exp_norm, n_dft, n_baseline):
    """Fit using lsq_linear: DFT coefficients >= 0, baseline unconstrained."""
    n_total = n_dft + n_baseline
    lb = np.zeros(n_total)
    ub = np.full(n_total, np.inf)
    # Baseline terms: unconstrained (allow negative)
    lb[n_dft:] = -np.inf
    result = lsq_linear(A, exp_norm, bounds=(lb, ub))
    coeffs = result.x
    dft_coeffs = coeffs[:n_dft]
    baseline_coeffs = coeffs[n_dft:]
    residual = exp_norm - A @ coeffs
    return dft_coeffs, baseline_coeffs, residual


def fit_subset_with_baseline(dft_matrix, subset_indices, exp_wn, exp_norm, poly_order=1):
    """Fit a subset of DFT spectra + baseline."""
    sub_matrix = dft_matrix[subset_indices]
    A, n_dft, n_baseline = build_design_matrix_with_baseline(sub_matrix, exp_wn, poly_order)
    dft_coeffs, baseline_coeffs, residual = fit_with_baseline(A, exp_norm, n_dft, n_baseline)
    rss = np.sum(residual**2)
    return dft_coeffs, baseline_coeffs, rss, residual


def pearson_smoothed_derivative(exp_norm, dft_matrix, exp_wn, sigma=2.0):
    """Rank by Pearson correlation on Gaussian-smoothed first derivatives.
    More discriminating than cosine on raw spectra."""
    # Smoothed first derivative of experimental
    exp_smooth = gaussian_filter1d(exp_norm, sigma=sigma)
    exp_deriv = np.gradient(exp_smooth, exp_wn)
    exp_deriv_centered = exp_deriv - exp_deriv.mean()
    exp_deriv_norm = np.linalg.norm(exp_deriv_centered)

    scores = []
    for i in range(dft_matrix.shape[0]):
        dft_smooth = gaussian_filter1d(dft_matrix[i], sigma=sigma)
        dft_deriv = np.gradient(dft_smooth, exp_wn)
        dft_deriv_centered = dft_deriv - dft_deriv.mean()
        dft_deriv_norm = np.linalg.norm(dft_deriv_centered)
        if exp_deriv_norm > 0 and dft_deriv_norm > 0:
            pearson = np.dot(exp_deriv_centered, dft_deriv_centered) / (
                exp_deriv_norm * dft_deriv_norm)
        else:
            pearson = 0.0
        scores.append(pearson)
    return np.array(scores)


def cosine_similarity(exp_norm, dft_matrix):
    """Standard cosine similarity (for comparison)."""
    scores = []
    norm_exp = np.linalg.norm(exp_norm)
    for i in range(dft_matrix.shape[0]):
        norm_dft = np.linalg.norm(dft_matrix[i])
        if norm_exp > 0 and norm_dft > 0:
            scores.append(np.dot(exp_norm, dft_matrix[i]) / (norm_exp * norm_dft))
        else:
            scores.append(0.0)
    return np.array(scores)


def forward_stepwise_selection(dft_matrix, exp_wn, exp_norm, max_k, poly_order=1):
    """Forward stepwise: at each step, add the component that maximally reduces RSS."""
    n_mol = dft_matrix.shape[0]
    available = set(range(n_mol))
    selected = []
    history = []  # list of (selected_indices, rss, delta_rss)

    for step in range(max_k):
        best_rss = np.inf
        best_idx = None
        for candidate in available:
            trial = selected + [candidate]
            _, _, rss, _ = fit_subset_with_baseline(
                dft_matrix, trial, exp_wn, exp_norm, poly_order)
            if rss < best_rss:
                best_rss = rss
                best_idx = candidate
        if best_idx is None:
            break
        selected.append(best_idx)
        available.remove(best_idx)
        prev_rss = history[-1][1] if history else np.sum(
            (exp_norm - np.mean(exp_norm))**2)  # baseline-only RSS
        delta_rss = prev_rss - best_rss
        history.append((list(selected), best_rss, delta_rss))

    return history


def exhaustive_subset_search(dft_matrix, exp_wn, exp_norm, k, poly_order=1):
    """Exhaustive search for best k-subset (only feasible for small k)."""
    from itertools import combinations
    n_mol = dft_matrix.shape[0]
    best_rss = np.inf
    best_subset = None

    for subset in combinations(range(n_mol), k):
        _, _, rss, _ = fit_subset_with_baseline(
            dft_matrix, list(subset), exp_wn, exp_norm, poly_order)
        if rss < best_rss:
            best_rss = rss
            best_subset = list(subset)

    return best_subset, best_rss


def compute_bic(rss, k, n_eff):
    """BIC with effective n (bandwidth-corrected)."""
    return n_eff * np.log(rss / n_eff) + k * np.log(n_eff)


def bootstrap_weights(dft_matrix, selected_indices, exp_wn, exp_norm,
                      n_bootstrap=500, poly_order=1):
    """Bootstrap: resample wavenumber blocks, refit, get CI on weights."""
    n_wn = len(exp_wn)
    # Block bootstrap: blocks of size ~ FELIX_FWHM / step
    step = exp_wn[1] - exp_wn[0] if len(exp_wn) > 1 else 1.0
    block_size = max(int(FELIX_FWHM / step), 3)
    n_blocks = n_wn // block_size

    all_weights = []
    sub_matrix = dft_matrix[selected_indices]

    for _ in range(n_bootstrap):
        # Resample blocks with replacement
        block_indices = np.random.randint(0, n_blocks, size=n_blocks)
        sample_idx = []
        for bi in block_indices:
            start = bi * block_size
            end = min(start + block_size, n_wn)
            sample_idx.extend(range(start, end))
        sample_idx = np.array(sample_idx[:n_wn])  # keep same length

        exp_boot = exp_norm[sample_idx]
        wn_boot = exp_wn[sample_idx]
        dft_boot = sub_matrix[:, sample_idx]

        A_boot, n_dft, n_base = build_design_matrix_with_baseline(
            dft_boot, wn_boot, poly_order)
        dft_c, _, _ = fit_with_baseline(A_boot, exp_boot, n_dft, n_base)

        total = dft_c.sum()
        if total > 0:
            all_weights.append(dft_c / total)
        else:
            all_weights.append(dft_c)

    all_weights = np.array(all_weights)
    return all_weights


def peak_resolved_residuals(exp_wn, exp_norm, reconstructed):
    """Compute residuals at experimental peak positions."""
    # Find peaks in experimental
    peaks, properties = find_peaks(exp_norm, height=0.1, distance=5, prominence=0.05)
    if len(peaks) == 0:
        return np.array([]), np.array([]), np.array([])
    peak_wn = exp_wn[peaks]
    peak_exp = exp_norm[peaks]
    peak_fit = reconstructed[peaks]
    peak_residuals = peak_exp - peak_fit
    return peak_wn, peak_residuals, peak_exp


# ============================================================
# MAIN ANALYSIS
# ============================================================
print("=" * 70)
print("  SPECTRAL ANALYSIS v2 — Methodologically Revised Pipeline")
print("=" * 70)

exp_wn, exp_norm = load_experimental()
molecules, dft_matrix = load_dft_spectra(exp_wn, DEFAULT_SCALE)
n_mol = len(molecules)
n_wn = len(exp_wn)
print(f"\nLoaded {n_mol} DFT spectra, {n_wn} wavenumber points.")
print(f"Effective independent points (n_eff): {N_EFF}")
print(f"Wavenumber step: {exp_wn[1]-exp_wn[0]:.1f} cm⁻¹")

# ============================================================
# DIAGNOSTICS: Collinearity
# ============================================================
print("\n" + "=" * 70)
print("DIAGNOSTICS: Collinearity of DFT Basis Set")
print("=" * 70)

A_raw = dft_matrix.T
cond_number = np.linalg.cond(A_raw)
print(f"  Condition number of DFT matrix: {cond_number:.2e}")
if cond_number > 1e4:
    print("  ⚠ HIGH collinearity — NNLS weights may be unstable between similar isomers")

# Gram matrix (pairwise DFT-DFT cosine)
norms = np.linalg.norm(dft_matrix, axis=1, keepdims=True)
norms[norms == 0] = 1.0
dft_normed = dft_matrix / norms
gram = dft_normed @ dft_normed.T
np.fill_diagonal(gram, 0)
max_cos = gram.max()
i_max, j_max = np.unravel_index(gram.argmax(), gram.shape)
print(f"  Max off-diagonal DFT-DFT cosine: {max_cos:.4f}")
print(f"    → between CID {molecules[i_max]['cid']} ({molecules[i_max]['name'][:30]})")
print(f"      and    CID {molecules[j_max]['cid']} ({molecules[j_max]['name'][:30]})")

# Count highly correlated pairs
high_corr_mask = gram > 0.90
n_high_corr = high_corr_mask.sum() // 2  # symmetric
print(f"  Pairs with cosine > 0.90: {n_high_corr}")
high_corr_mask_95 = gram > 0.95
n_very_high = high_corr_mask_95.sum() // 2
print(f"  Pairs with cosine > 0.95: {n_very_high}")

# ============================================================
# STEP 1: Ranking — Pearson on smoothed derivatives + Cosine
# ============================================================
print("\n" + "=" * 70)
print("STEP 1: Spectral Similarity Ranking")
print("=" * 70)

pearson_scores = pearson_smoothed_derivative(exp_norm, dft_matrix, exp_wn, sigma=2.0)
cosine_scores = cosine_similarity(exp_norm, dft_matrix)

pearson_ranking = np.argsort(pearson_scores)[::-1]
cosine_ranking = np.argsort(cosine_scores)[::-1]

print(f"\n{'Rank':<5} {'CID':<12} {'Name':<35} {'Pearson(∂)':>10} {'Cosine':>8}")
print("-" * 75)
for r in range(min(15, n_mol)):
    idx = pearson_ranking[r]
    print(f"{r+1:<5} {molecules[idx]['cid']:<12} {molecules[idx]['name'][:34]:<35} "
          f"{pearson_scores[idx]:>10.4f} {cosine_scores[idx]:>8.4f}")

# ============================================================
# STEP 2: Full NNLS with baseline (all 41)
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: NNLS with Baseline (all components)")
print("=" * 70)

A_full, n_dft_full, n_base_full = build_design_matrix_with_baseline(
    dft_matrix, exp_wn, poly_order=1)
dft_coeffs_full, baseline_coeffs_full, residual_full = fit_with_baseline(
    A_full, exp_norm, n_dft_full, n_base_full)

coeff_sum_full = dft_coeffs_full.sum()
weights_full = dft_coeffs_full / coeff_sum_full if coeff_sum_full > 0 else dft_coeffs_full
nnls_ranking_full = np.argsort(weights_full)[::-1]

ss_tot = np.sum((exp_norm - exp_norm.mean())**2)
ss_res_full = np.sum(residual_full**2)
r2_full = 1 - ss_res_full / ss_tot

non_zero_full = np.sum(dft_coeffs_full > 0)
exactly_zero = np.sum(dft_coeffs_full == 0)

print(f"  Baseline coefficients: const={baseline_coeffs_full[0]:.4f}, "
      f"linear={baseline_coeffs_full[1]:.4f}")
print(f"  R² (full model): {r2_full:.4f}")
print(f"  Non-zero DFT coefficients: {non_zero_full} / {n_mol}")
print(f"  Exactly zero (NNLS active set): {exactly_zero}")

print(f"\n{'Rank':<5} {'CID':<12} {'Name':<35} {'Spectral Wt':>11} {'Raw Coeff':>10}")
print("-" * 78)
for r, idx in enumerate(nnls_ranking_full):
    if weights_full[idx] < 1e-4:
        break
    print(f"{r+1:<5} {molecules[idx]['cid']:<12} {molecules[idx]['name'][:34]:<35} "
          f"{weights_full[idx]:>10.1%} {dft_coeffs_full[idx]:>10.5f}")

# ============================================================
# STEP 3: Forward Stepwise Selection
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Forward Stepwise Selection (RSS-optimal at each step)")
print("=" * 70)

stepwise_history = forward_stepwise_selection(
    dft_matrix, exp_wn, exp_norm, max_k=MAX_K_FORWARD, poly_order=1)

print(f"\n{'k':<4} {'Added CID':<12} {'Added Name':<35} {'RSS':>8} {'ΔRSS':>8} {'R²':>7}")
print("-" * 78)
for step, (sel, rss, drss) in enumerate(stepwise_history):
    added_idx = sel[-1]
    r2 = 1 - rss / ss_tot
    print(f"{step+1:<4} {molecules[added_idx]['cid']:<12} "
          f"{molecules[added_idx]['name'][:34]:<35} "
          f"{rss:>8.3f} {drss:>8.3f} {r2:>7.4f}")

# ============================================================
# STEP 4: BIC with n_eff (bandwidth-corrected)
# ============================================================
print("\n" + "=" * 70)
print(f"STEP 4: BIC Model Selection (n_eff = {N_EFF})")
print("=" * 70)

bic_values = []
for step, (sel, rss, _) in enumerate(stepwise_history):
    k = step + 1
    bic = compute_bic(rss, k, N_EFF)
    bic_values.append(bic)
    print(f"  k={k:2d}  BIC(n_eff) = {bic:8.2f}  RSS = {rss:.4f}")

best_k = np.argmin(bic_values) + 1
best_selection = stepwise_history[best_k - 1][0]
print(f"\n** Optimal k (lowest BIC with n_eff={N_EFF}): k = {best_k} **")

# ============================================================
# STEP 5: Exhaustive search for small k (if best_k <= MAX_K_EXHAUSTIVE)
# ============================================================
if best_k <= MAX_K_EXHAUSTIVE:
    print(f"\n" + "=" * 70)
    print(f"STEP 5a: Exhaustive Search for k={best_k}")
    print("=" * 70)
    from itertools import combinations
    n_combos = 1
    for i in range(best_k):
        n_combos = n_combos * (n_mol - i) // (i + 1)
    print(f"  Testing all C({n_mol},{best_k}) = {n_combos} subsets...")
    exh_subset, exh_rss = exhaustive_subset_search(
        dft_matrix, exp_wn, exp_norm, best_k, poly_order=1)
    print(f"  Exhaustive best subset RSS: {exh_rss:.4f}")
    print(f"  Forward stepwise RSS:       {stepwise_history[best_k-1][1]:.4f}")
    print(f"  Exhaustive selection:")
    for idx in exh_subset:
        print(f"    CID {molecules[idx]['cid']} — {molecules[idx]['name']}")
    # Use exhaustive result if better
    if exh_rss < stepwise_history[best_k - 1][1]:
        print("  → Using exhaustive result (better than stepwise)")
        best_selection = exh_subset
    else:
        print("  → Stepwise result matches exhaustive (or is same)")

# ============================================================
# STEP 6: Final fit with best_k + Bootstrap CI
# ============================================================
print("\n" + "=" * 70)
print(f"STEP 6: Final Fit (k={best_k}) + Bootstrap Confidence Intervals")
print("=" * 70)

dft_coeffs_best, baseline_best, rss_best, residual_best = fit_subset_with_baseline(
    dft_matrix, best_selection, exp_wn, exp_norm, poly_order=1)
recon_best = exp_norm - residual_best
r2_best = 1 - rss_best / ss_tot

weights_best = dft_coeffs_best / dft_coeffs_best.sum() if dft_coeffs_best.sum() > 0 else dft_coeffs_best

print(f"\n  Final R² = {r2_best:.4f}")
print(f"  Baseline: const={baseline_best[0]:.4f}, linear={baseline_best[1]:.4f}")

# Bootstrap
print(f"\n  Running {N_BOOTSTRAP} block-bootstrap iterations...")
boot_weights = bootstrap_weights(
    dft_matrix, best_selection, exp_wn, exp_norm,
    n_bootstrap=N_BOOTSTRAP, poly_order=1)

print(f"\n{'#':<3} {'CID':<12} {'Name':<35} {'Weight':>7} {'95% CI':>18}")
print("-" * 80)
for i, idx in enumerate(best_selection):
    w = weights_best[i]
    ci_lo = np.percentile(boot_weights[:, i], 2.5)
    ci_hi = np.percentile(boot_weights[:, i], 97.5)
    print(f"{i+1:<3} {molecules[idx]['cid']:<12} {molecules[idx]['name'][:34]:<35} "
          f"{w:>6.1%} [{ci_lo:>5.1%}, {ci_hi:>5.1%}]")

# Check for overlapping CIs (non-identifiable pairs)
print("\n  Identifiability check (overlapping 95% CIs):")
overlap_found = False
for i in range(len(best_selection)):
    for j in range(i+1, len(best_selection)):
        ci_i = (np.percentile(boot_weights[:, i], 2.5),
                np.percentile(boot_weights[:, i], 97.5))
        ci_j = (np.percentile(boot_weights[:, j], 2.5),
                np.percentile(boot_weights[:, j], 97.5))
        if ci_i[0] < ci_j[1] and ci_j[0] < ci_i[1]:
            overlap_found = True
            print(f"    ⚠ CID {molecules[best_selection[i]]['cid']} and "
                  f"CID {molecules[best_selection[j]]['cid']} have overlapping CIs")
if not overlap_found:
    print("    ✓ All components have non-overlapping CIs")

# ============================================================
# STEP 7: Peak-resolved residual analysis
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: Peak-Resolved Residual Analysis")
print("=" * 70)

peak_wn, peak_resid, peak_exp = peak_resolved_residuals(exp_wn, exp_norm, recon_best)
if len(peak_wn) > 0:
    print(f"\n  Detected {len(peak_wn)} experimental peaks (prominence > 0.05)")
    print(f"  Mean absolute peak residual: {np.mean(np.abs(peak_resid)):.4f}")
    print(f"  Max absolute peak residual:  {np.max(np.abs(peak_resid)):.4f}")
    print(f"\n  {'Peak cm⁻¹':>10} {'Exp Int':>8} {'Fit Int':>8} {'Residual':>9}")
    print("  " + "-" * 40)
    for pw, pr, pe in zip(peak_wn, peak_resid, peak_exp):
        fit_val = pe - pr
        print(f"  {pw:>10.1f} {pe:>8.3f} {fit_val:>8.3f} {pr:>+9.3f}")
else:
    print("  No peaks detected.")

# ============================================================
# STEP 8: Sensitivity to Scaling Factor
# ============================================================
print("\n" + "=" * 70)
print("STEP 8: Sensitivity to DFT Scaling Factor")
print("=" * 70)

sensitivity_results = []
print(f"\n  {'Scale':<7} {'Best k':<7} {'R²':>7} {'BIC':>9} {'Top CID':<12} {'Top Wt':>7}")
print("  " + "-" * 55)

for sf in SCALING_FACTORS:
    # Need to reload with different scaling — but files only exist for 0.95
    # So we SHIFT the wavenumber grid instead: unscaled_wn = dft_wn / 0.95 * sf
    # This is equivalent to changing the scale factor
    shift_ratio = sf / DEFAULT_SCALE
    dft_matrix_shifted = np.zeros_like(dft_matrix)
    for i in range(n_mol):
        # Shift DFT wavenumber grid and re-interpolate
        shifted_wn = exp_wn * shift_ratio
        dft_matrix_shifted[i] = np.interp(exp_wn, shifted_wn, dft_matrix[i],
                                          left=0, right=0)

    # Quick forward stepwise with this shifted matrix
    hist_sf = forward_stepwise_selection(
        dft_matrix_shifted, exp_wn, exp_norm, max_k=8, poly_order=1)

    bic_sf = [compute_bic(rss, k+1, N_EFF) for k, (_, rss, _) in enumerate(hist_sf)]
    best_k_sf = np.argmin(bic_sf) + 1
    best_sel_sf = hist_sf[best_k_sf - 1][0]
    best_rss_sf = hist_sf[best_k_sf - 1][1]
    r2_sf = 1 - best_rss_sf / ss_tot

    # Get top component
    dft_c_sf, _, _, _ = fit_subset_with_baseline(
        dft_matrix_shifted, best_sel_sf, exp_wn, exp_norm, poly_order=1)
    top_idx_sf = best_sel_sf[np.argmax(dft_c_sf)]

    sensitivity_results.append({
        "scale": sf, "best_k": best_k_sf, "r2": r2_sf,
        "bic": bic_sf[best_k_sf - 1],
        "top_cid": molecules[top_idx_sf]["cid"],
        "top_weight": dft_c_sf.max() / dft_c_sf.sum() if dft_c_sf.sum() > 0 else 0,
        "selection": best_sel_sf
    })
    print(f"  {sf:<7.2f} {best_k_sf:<7d} {r2_sf:>7.4f} {bic_sf[best_k_sf-1]:>9.2f} "
          f"{molecules[top_idx_sf]['cid']:<12} {dft_c_sf.max()/dft_c_sf.sum() if dft_c_sf.sum()>0 else 0:>6.1%}")

# Check if top assignment is stable
top_cids_across_scales = [r["top_cid"] for r in sensitivity_results]
if len(set(top_cids_across_scales)) == 1:
    print(f"\n  ✓ Top assignment stable across scaling factors: CID {top_cids_across_scales[0]}")
else:
    print(f"\n  ⚠ Top assignment varies with scaling: {top_cids_across_scales}")

# ============================================================
# OUTPUT: Summary CSV
# ============================================================
summary_data = []
for i in range(n_mol):
    summary_data.append({
        "CID": molecules[i]["cid"],
        "Name": molecules[i]["name"],
        "Pearson_Derivative": pearson_scores[i],
        "Cosine_Similarity": cosine_scores[i],
        "NNLS_Spectral_Weight_Percent": weights_full[i] * 100,
        "Pearson_Rank": int(np.where(pearson_ranking == i)[0][0]) + 1,
        "Cosine_Rank": int(np.where(cosine_ranking == i)[0][0]) + 1,
        "NNLS_Rank": int(np.where(nnls_ranking_full == i)[0][0]) + 1,
        "In_Best_Model": "Yes" if i in best_selection else "No",
    })

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values("NNLS_Spectral_Weight_Percent", ascending=False).reset_index(drop=True)
summary_csv = os.path.join(OUTPUT_DIR, "140_spectral_analysis_v2_summary.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"\nSummary CSV saved to: {summary_csv}")

# ============================================================
# OUTPUT: PDF Report
# ============================================================
output_pdf = os.path.join(OUTPUT_DIR, "140_spectral_analysis_v2.pdf")

with PdfPages(output_pdf) as pdf:

    # --- Page 1: Ranking comparison ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    top_n = min(15, n_mol)

    # Pearson derivative ranking
    top_p = pearson_ranking[:top_n]
    ax1.barh(range(top_n), pearson_scores[top_p], color="teal", edgecolor="none")
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels([f"CID {molecules[i]['cid']}" for i in top_p], fontsize=7)
    ax1.set_xlabel("Pearson (smoothed derivative)")
    ax1.set_title("Ranking: Pearson on ∂/∂ν (smoothed)", fontweight="bold", fontsize=9)
    ax1.invert_yaxis()

    # Cosine ranking
    top_c = cosine_ranking[:top_n]
    ax2.barh(range(top_n), cosine_scores[top_c], color="steelblue", edgecolor="none")
    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels([f"CID {molecules[i]['cid']}" for i in top_c], fontsize=7)
    ax2.set_xlabel("Cosine Similarity")
    ax2.set_title("Ranking: Cosine (raw spectra)", fontweight="bold", fontsize=9)
    ax2.invert_yaxis()

    plt.tight_layout()
    pdf.savefig(fig, dpi=100)
    plt.close(fig)

    # --- Page 2: Forward stepwise R² + BIC ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ks = range(1, len(stepwise_history) + 1)
    r2s = [1 - h[1] / ss_tot for h in stepwise_history]
    ax1.plot(ks, r2s, "o-", color="darkgreen", lw=2, markersize=5)
    ax1.axvline(x=best_k, color="crimson", ls="--", lw=1, label=f"BIC optimal k={best_k}")
    ax1.set_xlabel("Number of Components (k)")
    ax1.set_ylabel("R²")
    ax1.set_title("Forward Stepwise: Cumulative R²", fontweight="bold")
    ax1.legend()
    ax1.set_ylim(bottom=0)

    ax2.plot(ks, bic_values, "s-", color="crimson", lw=2, markersize=5)
    ax2.axvline(x=best_k, color="gray", ls="--", lw=1, label=f"Optimal k={best_k}")
    ax2.set_xlabel("Number of Components (k)")
    ax2.set_ylabel(f"BIC (n_eff={N_EFF})")
    ax2.set_title("BIC Model Selection (bandwidth-corrected)", fontweight="bold")
    ax2.legend()

    plt.tight_layout()
    pdf.savefig(fig, dpi=100)
    plt.close(fig)

    # --- Page 3: Best fit overlay + residual ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7),
                                    gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(exp_wn, exp_norm, "k-", lw=1.2, label="Experimental m/z 140")
    ax1.plot(exp_wn, recon_best, "r-", lw=1.0, alpha=0.85,
             label=f"NNLS fit (k={best_k}, R²={r2_best:.3f})")

    colors = plt.colormaps["tab10"]
    for i, idx in enumerate(best_selection):
        contribution = dft_matrix[idx] * dft_coeffs_best[i]
        ax1.fill_between(exp_wn, 0, contribution, alpha=0.2, color=colors(i % 10),
                         label=f"CID {molecules[idx]['cid']} ({weights_best[i]:.0%})")

    ax1.set_xlim(WAVENUMBER_RANGE)
    ax1.set_ylabel("Intensity (norm.)")
    ax1.set_title(f"Best Fit: {best_k}-Component Decomposition (spectral weights, not populations)",
                  fontweight="bold", fontsize=9)
    ax1.legend(fontsize=6, loc="upper right", ncol=2)

    ax2.plot(exp_wn, residual_best, "k-", lw=0.8)
    ax2.axhline(0, color="gray", ls="--", lw=0.5)
    ax2.fill_between(exp_wn, 0, residual_best, alpha=0.3, color="gray")
    if len(peak_wn) > 0:
        ax2.scatter(peak_wn, peak_resid, c="red", s=15, zorder=5, label="Peak residuals")
        ax2.legend(fontsize=7)
    ax2.set_xlim(WAVENUMBER_RANGE)
    ax2.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax2.set_ylabel("Residual")
    ax2.set_title("Residual (Exp − Fit) with peak positions marked")

    plt.tight_layout()
    pdf.savefig(fig, dpi=100)
    plt.close(fig)

    # --- Page 4: Bootstrap weight distributions ---
    fig, ax = plt.subplots(figsize=(10, 5))
    positions = range(len(best_selection))
    bp = ax.boxplot([boot_weights[:, i] * 100 for i in range(len(best_selection))],
                    positions=positions, vert=True, patch_artist=True,
                    medianprops=dict(color="black", lw=1.5))
    for patch, i in zip(bp["boxes"], range(len(best_selection))):
        patch.set_facecolor(colors(i % 10))
        patch.set_alpha(0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"CID {molecules[idx]['cid']}" for idx in best_selection],
                       fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Spectral Weight (%)")
    ax.set_title(f"Bootstrap Distributions (n={N_BOOTSTRAP}, block resampling)",
                 fontweight="bold")
    plt.tight_layout()
    pdf.savefig(fig, dpi=100)
    plt.close(fig)

    # --- Page 5: Sensitivity to scaling factor ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for sr in sensitivity_results:
        sf = sr["scale"]
        sel = sr["selection"]
        dft_shifted = np.zeros_like(dft_matrix)
        shift_ratio = sf / DEFAULT_SCALE
        for i in range(n_mol):
            shifted_wn = exp_wn * shift_ratio
            dft_shifted[i] = np.interp(exp_wn, shifted_wn, dft_matrix[i], left=0, right=0)
        dft_c, _, _, _ = fit_subset_with_baseline(
            dft_shifted, sel, exp_wn, exp_norm, poly_order=1)
        recon_sf = exp_norm - (exp_norm - (
            build_design_matrix_with_baseline(dft_shifted[sel], exp_wn, 1)[0] @
            np.concatenate([dft_c, fit_subset_with_baseline(dft_shifted, sel, exp_wn, exp_norm, 1)[1]])))
        # Simpler: just plot R² vs scale
    scales = [r["scale"] for r in sensitivity_results]
    r2s_sf = [r["r2"] for r in sensitivity_results]
    ax.plot(scales, r2s_sf, "o-", color="purple", lw=2, markersize=8)
    ax.set_xlabel("DFT Scaling Factor")
    ax.set_ylabel("R² (best model)")
    ax.set_title("Sensitivity: R² vs. Scaling Factor", fontweight="bold")
    for sr in sensitivity_results:
        ax.annotate(f"k={sr['best_k']}", (sr["scale"], sr["r2"]),
                    textcoords="offset points", xytext=(0, 10), fontsize=8, ha="center")
    plt.tight_layout()
    pdf.savefig(fig, dpi=100)
    plt.close(fig)

    # --- Page 6: DFT-DFT Gram matrix heatmap ---
    fig, ax = plt.subplots(figsize=(10, 9))
    gram_full = dft_normed @ dft_normed.T
    im = ax.imshow(gram_full, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n_mol))
    ax.set_yticks(range(n_mol))
    labels = [f"{molecules[i]['cid']}" for i in range(n_mol)]
    ax.set_xticklabels(labels, fontsize=5, rotation=90)
    ax.set_yticklabels(labels, fontsize=5)
    ax.set_title("DFT-DFT Cosine Similarity (Gram Matrix)", fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    pdf.savefig(fig, dpi=100)
    plt.close(fig)

print(f"\nPDF report saved to: {output_pdf}")
print("\n" + "=" * 70)
print("  IMPORTANT CAVEATS")
print("=" * 70)
print("""
  • Spectral weights ≠ populations. IRMPD is a nonlinear, multi-photon
    process. Different isomers may have different IRMPD efficiencies.
    Weights represent relative contribution to the *spectrum*, not
    relative abundance in the ion population.

  • To convert to populations, independent validation is required
    (e.g., NMR, isomer-selected experiments, IR-IR double resonance).

  • Collinearity between DFT spectra limits identifiability. If two
    isomers have cosine > 0.95, their individual weights are unreliable
    — only their combined contribution is robust.

  • DFT harmonic frequencies (even scaled) may not match IRMPD peak
    positions exactly due to anharmonicity, temperature effects, and
    the multi-photon nature of IRMPD dissociation.
""")
print("Done!")
