"""
Spectral Analysis Pipeline v3 — Comprehensive Revision
=======================================================
Compares experimental m/z 140 IR-UV ion-dip spectrum against
B3LYP/def2-TZVP anharmonic DFT spectra (41 isomer candidates).

IMPORTANT CAVEAT:
  The UV laser was parked at the wavelength of maximum m/z 140 ion yield.
  Under this geometry, recovered spectral weights reflect contributions
  weighted by the product: n_i * sigma_UV(i)(nu_UV) * phi_ion(i).
  They are NOT isomer populations. Isomers whose S1 origin lies far from
  the chosen UV wavelength contribute zero signal regardless of abundance.
  Absence claims cannot be made for UV-silent candidates.

  To convert spectral weights to populations, independent calibration is
  required (e.g., R2PI action spectra per isomer, NMR, isomer-selected
  experiments, or IR-IR double resonance).

Pipeline:
  0. Diagnostics (collinearity, condition number, DFT-DFT Gram matrix)
  1. Ranking: Pearson on smoothed first-derivative spectra (+ cosine)
  2. NNLS with explicit baseline term via lsq_linear (mixed bounds)
  3. Forward stepwise subset selection (RSS-optimal at each step)
     + exhaustive search for small k
  4. Model selection: BIC(n_eff) + blocked k-fold cross-validation
  5. Final fit with optimal k
  6. Block bootstrap CIs on spectral weights (selection frequency,
     pairwise rank stability)
  7. Peak-resolved residual analysis
  8. Sensitivity to DFT scaling factor

References:
  - NNLS: Lawson & Hanson (1974) "Solving Least Squares Problems"
  - IR mixture decomposition: Barnes et al. (2020) IJMS 447, 116235
  - Large-scale NNLS mixture ID: Martynko et al. (2025) arXiv:2602.21308
  - BIC: Schwarz (1978) Ann. Statist. 6(2), 461-464
  - Cosine scoring: Stein & Scott (1994) JASMS 5, 859-866
"""

import os
import sys
import glob
import re
import datetime
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import nnls, lsq_linear
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter
from scipy.cluster.hierarchy import linkage, fcluster
from matplotlib.image import imread
from matplotlib.gridspec import GridSpec

# ================================================================
# CONFIGURATION — all tuneable constants in one place
# ================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_FILE = os.path.join(BASE_DIR, "mass_channel_IR_spectra_wn401-1576_2ch.csv")
EXP_RAW_FILE = os.path.join(BASE_DIR, "mass_channel_IR_spectra_wn401-1576_1ch.csv")
DFT_DIR = os.path.join(BASE_DIR, "b3lyp_def2-tzvp", "scaled")
IMG_DIR = os.path.join(BASE_DIR, "PubChem_search_images-2")
OUTPUT_DIR = os.path.join(BASE_DIR, "b3lyp_def2-tzvp", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

WAVENUMBER_RANGE = (400, 1600)          # cm-1, display/analysis range
FWHM_CM = 10.0                         # approximate experimental bandwidth (cm-1)
SCALING_FACTOR = 0.95                   # DFT frequency scaling factor used
SCALING_SENSITIVITY = [0.94, 0.95, 0.96, 0.97]
NONZERO_THRESHOLD = 0.0                # use exact NNLS active-set zeros
N_BOOTSTRAP = 1000                     # bootstrap iterations
MAX_K_EXHAUSTIVE = 5                   # exhaustive subset search up to this k
MAX_K_FORWARD = 12                     # forward stepwise limit
N_CV_BLOCKS = 10                       # blocks for k-fold CV
POLY_ORDER = 1                         # baseline polynomial order (0=const, 1=linear)

# Savitzky-Golay smoothing sensitivity: (window_length, polyorder) pairs
SG_PARAMS = [
    (5, 2),    # minimal smoothing
    (9, 2),    # light smoothing
    (15, 3),   # moderate smoothing
    (21, 3),   # heavier smoothing
    (31, 3),   # strong smoothing
    (41, 4),   # very strong smoothing
]

SCRIPT_NAME = os.path.basename(__file__)
RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
RUN_TIMESTAMP_FILE = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Effective independent spectral points (bandwidth-corrected)
WN_SPAN = WAVENUMBER_RANGE[1] - WAVENUMBER_RANGE[0]
N_EFF = int(WN_SPAN / FWHM_CM)

# ================================================================
# HELPER FUNCTIONS
# ================================================================

def load_experimental():
    exp_df = pd.read_csv(EXP_FILE)
    exp_wn = exp_df["Wavenumber"].values
    exp_int = exp_df["m/z 140.0 -ln(depl)"].values
    mask = (exp_wn >= WAVENUMBER_RANGE[0]) & (exp_wn <= WAVENUMBER_RANGE[1])
    exp_wn = exp_wn[mask]
    exp_int = exp_int[mask]
    # Normalize by max (robust: use max, not 99th pct, since ASPLS-corrected)
    exp_max = exp_int.max()
    exp_norm = exp_int / exp_max if exp_max > 0 else exp_int
    return exp_wn, exp_norm, exp_max


def load_dft_spectra(exp_wn, scale_factor=0.95):
    pattern = f"CID_*_scaled_{scale_factor:.2f}.csv"
    conv_files = sorted(glob.glob(os.path.join(DFT_DIR, pattern)))
    molecules, dft_list = [], []
    for cf in conv_files:
        bn = os.path.basename(cf)
        m = re.match(rf"CID_(\d+)_(.+)_scaled_{scale_factor:.2f}\.csv", bn)
        if not m:
            continue
        cid, name = m.group(1), m.group(2).replace("_", " ")
        df = pd.read_csv(cf, comment="#", header=None,
                         names=["Wavenumber", "Fundamentals", "Overtones",
                                "Combinations", "Total"])
        dft_wn = df["Wavenumber"].values
        dft_tot = df["Total"].values
        # Coverage check
        if dft_wn.min() > WAVENUMBER_RANGE[0] or dft_wn.max() < WAVENUMBER_RANGE[1]:
            print(f"  ⚠ CID {cid} DFT range [{dft_wn.min():.0f}, {dft_wn.max():.0f}] "
                  f"does not fully cover [{WAVENUMBER_RANGE[0]}, {WAVENUMBER_RANGE[1]}]")
        dft_interp = np.interp(exp_wn, dft_wn, dft_tot, left=0.0, right=0.0)
        molecules.append({"cid": cid, "name": name, "file": cf})
        dft_list.append(dft_interp)
    return molecules, np.array(dft_list)


def build_A(dft_sub, exp_wn, poly_order=POLY_ORDER):
    """Design matrix: [DFT columns | polynomial baseline columns]."""
    n_wn = len(exp_wn)
    wn_c = 2 * (exp_wn - exp_wn.min()) / (exp_wn.max() - exp_wn.min()) - 1
    base_cols = [wn_c**p for p in range(poly_order + 1)]
    B = np.column_stack(base_cols)
    A = np.hstack([dft_sub.T, B])
    return A, dft_sub.shape[0], B.shape[1]


def fit_lsq(dft_sub, exp_wn, exp_norm, poly_order=POLY_ORDER):
    """lsq_linear: DFT coeffs >= 0, baseline unconstrained."""
    A, nd, nb = build_A(dft_sub, exp_wn, poly_order)
    lb = np.concatenate([np.zeros(nd), np.full(nb, -np.inf)])
    ub = np.full(nd + nb, np.inf)
    res = lsq_linear(A, exp_norm, bounds=(lb, ub), method="bvls")
    dc, bc = res.x[:nd], res.x[nd:]
    resid = exp_norm - A @ res.x
    rss = np.sum(resid**2)
    return dc, bc, rss, resid


def felix_smooth(spectrum, wn_axis, bw_frac=0.0053):
    """Apply frequency-dependent Gaussian smoothing mimicking FELIX lineshape.
    FWHM at each point = bw_frac * wavenumber. Implemented via variable-width
    convolution (point-by-point weighted average)."""
    n = len(spectrum)
    smoothed = np.zeros(n)
    for i in range(n):
        fwhm_i = bw_frac * wn_axis[i]
        sigma_i = fwhm_i / (2 * np.sqrt(2 * np.log(2)))  # FWHM to Gaussian sigma
        # Kernel window: ±3 sigma in wavenumber units
        dists = np.abs(wn_axis - wn_axis[i])
        kernel = np.exp(-0.5 * (dists / sigma_i)**2)
        kernel /= kernel.sum()
        smoothed[i] = np.dot(kernel, spectrum)
    return smoothed


def pearson_deriv_scores(exp_norm, dft_matrix, exp_wn, bw_frac=0.0053):
    """Pearson on first derivatives. DFT smoothed with FELIX lineshape
    (frequency-dependent Gaussian, FWHM = 0.53% of ν). Exp used as-is."""
    ed = np.gradient(exp_norm, exp_wn)
    ed_c = ed - ed.mean()
    ed_n = np.linalg.norm(ed_c)
    scores = []
    for i in range(dft_matrix.shape[0]):
        ds = felix_smooth(dft_matrix[i], exp_wn, bw_frac)
        dd = np.gradient(ds, exp_wn)
        dd_c = dd - dd.mean()
        dd_n = np.linalg.norm(dd_c)
        if ed_n > 0 and dd_n > 0:
            scores.append(np.dot(ed_c, dd_c) / (ed_n * dd_n))
        else:
            scores.append(0.0)
    return np.array(scores)


def cosine_scores(exp_norm, dft_matrix):
    ne = np.linalg.norm(exp_norm)
    return np.array([np.dot(exp_norm, d) / (ne * np.linalg.norm(d))
                     if np.linalg.norm(d) > 0 else 0.0 for d in dft_matrix])


def forward_stepwise(dft_matrix, exp_wn, exp_norm, max_k):
    """At each step, add the candidate that maximally reduces RSS."""
    available = set(range(dft_matrix.shape[0]))
    selected = []
    history = []
    for _ in range(max_k):
        best_rss, best_i = np.inf, None
        for cand in available:
            trial = selected + [cand]
            _, _, rss, _ = fit_lsq(dft_matrix[trial], exp_wn, exp_norm)
            if rss < best_rss:
                best_rss, best_i = rss, cand
        if best_i is None:
            break
        selected.append(best_i)
        available.remove(best_i)
        history.append((list(selected), best_rss))
    return history


def exhaustive_search(dft_matrix, exp_wn, exp_norm, k):
    from itertools import combinations
    best_rss, best_sub = np.inf, None
    for sub in combinations(range(dft_matrix.shape[0]), k):
        _, _, rss, _ = fit_lsq(dft_matrix[list(sub)], exp_wn, exp_norm)
        if rss < best_rss:
            best_rss, best_sub = rss, list(sub)
    return best_sub, best_rss


def bic_neff(rss, k, n_eff):
    return n_eff * np.log(rss / n_eff) + k * np.log(n_eff)


def blocked_cv(dft_matrix, selected, exp_wn, exp_norm, n_blocks=N_CV_BLOCKS):
    """Blocked k-fold CV: hold out contiguous wavenumber blocks."""
    n = len(exp_norm)
    edges = np.linspace(0, n, n_blocks + 1, dtype=int)
    cv_err = 0.0
    for b in range(n_blocks):
        test_mask = np.zeros(n, dtype=bool)
        test_mask[edges[b]:edges[b+1]] = True
        train_mask = ~test_mask
        sub = dft_matrix[selected]
        A_tr, nd, nb = build_A(sub[:, train_mask], exp_wn[train_mask])
        lb = np.concatenate([np.zeros(nd), np.full(nb, -np.inf)])
        ub = np.full(nd + nb, np.inf)
        res = lsq_linear(A_tr, exp_norm[train_mask], bounds=(lb, ub), method="bvls")
        # Predict on test
        A_te, _, _ = build_A(sub[:, test_mask], exp_wn[test_mask])
        pred = A_te @ res.x
        cv_err += np.sum((exp_norm[test_mask] - pred)**2)
    return cv_err


def block_bootstrap_indices(n, block_len, rng):
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, max(n - block_len + 1, 1), size=n_blocks)
    idx = np.concatenate([np.arange(s, min(s + block_len, n)) for s in starts])
    return idx[:n]


def run_bootstrap(dft_matrix, selected, exp_wn, exp_norm, n_boot, block_len):
    rng = np.random.default_rng(42)
    n_sel = len(selected)
    all_weights = np.zeros((n_boot, n_sel))
    sub = dft_matrix[selected]
    for b in range(n_boot):
        idx = block_bootstrap_indices(len(exp_norm), block_len, rng)
        dc, _, _, _ = fit_lsq(sub[:, idx], exp_wn[idx], exp_norm[idx])
        s = dc.sum()
        all_weights[b] = dc / s if s > 0 else dc
    return all_weights


# ================================================================
# MAIN
# ================================================================
print("=" * 72)
print(f"  SPECTRAL ANALYSIS v3 — {SCRIPT_NAME}")
print(f"  Run: {RUN_TIMESTAMP}")
print("=" * 72)
print(f"\n  Config:")
print(f"    Wavenumber range : {WAVENUMBER_RANGE} cm⁻¹")
print(f"    FWHM             : {FWHM_CM} cm⁻¹")
print(f"    Scaling factor   : {SCALING_FACTOR}")
print(f"    n_eff            : {N_EFF}")
print(f"    Bootstrap iters  : {N_BOOTSTRAP}")
print(f"    Baseline poly    : order {POLY_ORDER}")

# --- Load data ---
exp_wn, exp_norm, exp_max_raw = load_experimental()
molecules, dft_matrix = load_dft_spectra(exp_wn, SCALING_FACTOR)
n_mol = len(molecules)
n_wn = len(exp_wn)
wn_step = exp_wn[1] - exp_wn[0] if n_wn > 1 else 1.0
block_len = max(int(3 * FWHM_CM / wn_step), 5)

print(f"\n  Data:")
print(f"    Experimental points : {n_wn}  (step {wn_step:.1f} cm⁻¹)")
print(f"    DFT candidates      : {n_mol}")
print(f"    Bootstrap block len : {block_len} points ({block_len*wn_step:.0f} cm⁻¹)")

# Log candidate list
print(f"\n  Candidate list:")
for i, mol in enumerate(molecules):
    print(f"    {i+1:>3}. CID {mol['cid']:<12} {mol['name'][:45]}")

ss_tot = np.sum((exp_norm - exp_norm.mean())**2)

# ================================================================
# DIAGNOSTICS: Collinearity
# ================================================================
print("\n" + "=" * 72)
print("DIAGNOSTICS: Collinearity")
print("=" * 72)

cond_num = np.linalg.cond(dft_matrix.T)
norms = np.linalg.norm(dft_matrix, axis=1, keepdims=True)
norms[norms == 0] = 1.0
dft_normed = dft_matrix / norms
gram = dft_normed @ dft_normed.T
np.fill_diagonal(gram, 0)
max_cos = gram.max()
i_mc, j_mc = np.unravel_index(gram.argmax(), gram.shape)

print(f"  cond(A)                = {cond_num:.2e}")
if cond_num > 1e4:
    print("  ⚠ HIGH — weights may be unstable between similar isomers")
print(f"  max |DFT_i · DFT_j|   = {max_cos:.4f}")
print(f"    → CID {molecules[i_mc]['cid']} vs CID {molecules[j_mc]['cid']}")

n_95 = (gram > 0.95).sum() // 2
n_90 = (gram > 0.90).sum() // 2
print(f"  Pairs with cos > 0.95 : {n_95}")
print(f"  Pairs with cos > 0.90 : {n_90}")

# Hierarchical clustering
dist = 1 - np.abs(gram + np.eye(n_mol))  # restore diagonal
condensed = dist[np.triu_indices(n_mol, k=1)]
Z = linkage(condensed, method="average")
cluster_labels = fcluster(Z, t=0.10, criterion="distance")
n_clusters = len(set(cluster_labels))
print(f"\n  Hierarchical clustering (cut=0.10): {n_clusters} clusters from {n_mol} isomers")
for cl in sorted(set(cluster_labels)):
    members = [i for i in range(n_mol) if cluster_labels[i] == cl]
    if len(members) > 1:
        cids = ", ".join(molecules[m]["cid"] for m in members)
        print(f"    Cluster {cl} ({len(members)} members): CIDs {cids}")

# ================================================================
# STEP 1: Ranking
# ================================================================
print("\n" + "=" * 72)
print("STEP 1: Similarity Ranking")
print("=" * 72)

pearson_sc = pearson_deriv_scores(exp_norm, dft_matrix, exp_wn)
cosine_sc = cosine_scores(exp_norm, dft_matrix)
pearson_rank = np.argsort(pearson_sc)[::-1]
cosine_rank = np.argsort(cosine_sc)[::-1]

print(f"\n{'Rank':<5} {'CID':<12} {'Name':<35} {'Pearson(∂)':>10} {'Cosine':>8}")
print("-" * 75)
for r in range(min(15, n_mol)):
    idx = pearson_rank[r]
    print(f"{r+1:<5} {molecules[idx]['cid']:<12} {molecules[idx]['name'][:34]:<35} "
          f"{pearson_sc[idx]:>10.4f} {cosine_sc[idx]:>8.4f}")

# ================================================================
# STEP 2: Full NNLS with baseline
# ================================================================
print("\n" + "=" * 72)
print("STEP 2: Full NNLS with Baseline (all candidates)")
print("=" * 72)

dc_full, bc_full, rss_full, resid_full = fit_lsq(dft_matrix, exp_wn, exp_norm)
r2_full = 1 - rss_full / ss_tot
wt_full = dc_full / dc_full.sum() if dc_full.sum() > 0 else dc_full
nnls_rank = np.argsort(wt_full)[::-1]
n_nonzero = int(np.sum(dc_full > NONZERO_THRESHOLD))

print(f"  Baseline: const={bc_full[0]:.5f}" +
      (f", linear={bc_full[1]:.5f}" if POLY_ORDER >= 1 else ""))
print(f"  R² (full model)       : {r2_full:.4f}")
print(f"  Non-zero (active set) : {n_nonzero} / {n_mol}")

print(f"\n{'Rank':<5} {'CID':<12} {'Name':<35} {'Spectral Wt':>11} {'Coeff':>9}")
print("-" * 76)
for r, idx in enumerate(nnls_rank):
    if dc_full[idx] <= NONZERO_THRESHOLD:
        break
    print(f"{r+1:<5} {molecules[idx]['cid']:<12} {molecules[idx]['name'][:34]:<35} "
          f"{wt_full[idx]:>10.1%} {dc_full[idx]:>9.5f}")

# ================================================================
# STEP 3: Forward Stepwise Selection
# ================================================================
print("\n" + "=" * 72)
print("STEP 3: Forward Stepwise Selection (RSS-optimal)")
print("=" * 72)

print("  Running forward stepwise (this may take a minute)...")
sw_history = forward_stepwise(dft_matrix, exp_wn, exp_norm, MAX_K_FORWARD)

print(f"\n{'k':<4} {'Added CID':<12} {'Name':<35} {'RSS':>8} {'R²':>7}")
print("-" * 70)
for step, (sel, rss) in enumerate(sw_history):
    added = sel[-1]
    r2 = 1 - rss / ss_tot
    print(f"{step+1:<4} {molecules[added]['cid']:<12} "
          f"{molecules[added]['name'][:34]:<35} {rss:>8.3f} {r2:>7.4f}")

# ================================================================
# STEP 4: Model Selection — BIC(n_eff) + Blocked CV
# ================================================================
print("\n" + "=" * 72)
print(f"STEP 4: Model Selection — BIC(n_eff={N_EFF}) + {N_CV_BLOCKS}-fold Blocked CV")
print("=" * 72)

bic_vals, cv_vals = [], []
for step, (sel, rss) in enumerate(sw_history):
    k = step + 1
    bv = bic_neff(rss, k, N_EFF)
    bic_vals.append(bv)
    cve = blocked_cv(dft_matrix, sel, exp_wn, exp_norm, N_CV_BLOCKS)
    cv_vals.append(cve)
    print(f"  k={k:2d}  BIC={bv:>9.2f}  CV_err={cve:>8.4f}  RSS={rss:>8.4f}")

best_k_bic = np.argmin(bic_vals) + 1
best_k_cv = np.argmin(cv_vals) + 1
print(f"\n  Optimal k (BIC n_eff) : {best_k_bic}")
print(f"  Optimal k (blocked CV): {best_k_cv}")

# Use CV-selected k as primary (more robust to autocorrelation)
best_k = best_k_cv
best_sel_sw = sw_history[best_k - 1][0]
print(f"  → Using blocked-CV selection: k = {best_k}")

# ================================================================
# STEP 5: Exhaustive Search (if feasible)
# ================================================================
if best_k <= MAX_K_EXHAUSTIVE:
    print(f"\n" + "=" * 72)
    print(f"STEP 5: Exhaustive Search for k={best_k}")
    print("=" * 72)
    from itertools import combinations
    n_combos = 1
    for i in range(best_k):
        n_combos = n_combos * (n_mol - i) // (i + 1)
    print(f"  Testing C({n_mol},{best_k}) = {n_combos} subsets...")
    exh_sub, exh_rss = exhaustive_search(dft_matrix, exp_wn, exp_norm, best_k)
    sw_rss = sw_history[best_k - 1][1]
    print(f"  Exhaustive best RSS : {exh_rss:.4f}")
    print(f"  Stepwise RSS        : {sw_rss:.4f}")
    for idx in exh_sub:
        print(f"    CID {molecules[idx]['cid']} — {molecules[idx]['name']}")
    if exh_rss < sw_rss - 1e-6:
        print("  → Exhaustive result is better; using it")
        best_sel = exh_sub
    else:
        print("  → Stepwise matched exhaustive")
        best_sel = best_sel_sw
else:
    print(f"\n  k={best_k} > {MAX_K_EXHAUSTIVE}, skipping exhaustive (using stepwise result)")
    best_sel = best_sel_sw

# ================================================================
# STEP 6: Final Fit + Bootstrap
# ================================================================
print("\n" + "=" * 72)
print(f"STEP 6: Final Fit (k={best_k}) + Bootstrap ({N_BOOTSTRAP} iterations)")
print("=" * 72)

dc_best, bc_best, rss_best, resid_best = fit_lsq(
    dft_matrix[best_sel], exp_wn, exp_norm)
recon_best = exp_norm - resid_best
r2_best = 1 - rss_best / ss_tot
wt_best = dc_best / dc_best.sum() if dc_best.sum() > 0 else dc_best

print(f"\n  R² = {r2_best:.4f}")
print(f"  Baseline: const={bc_best[0]:.5f}" +
      (f", linear={bc_best[1]:.5f}" if POLY_ORDER >= 1 else ""))

print(f"\n  Running block bootstrap (block_len={block_len})...")
boot_w = run_bootstrap(dft_matrix, best_sel, exp_wn, exp_norm, N_BOOTSTRAP, block_len)

# Selection frequency: how often each component gets non-zero weight
sel_freq = np.mean(boot_w > 1e-4, axis=0)

print(f"\n{'#':<3} {'CID':<12} {'Name':<32} {'Wt':>6} {'95% CI':>17} {'Sel%':>5}")
print("-" * 80)
for i, idx in enumerate(best_sel):
    lo = np.percentile(boot_w[:, i], 2.5)
    hi = np.percentile(boot_w[:, i], 97.5)
    print(f"{i+1:<3} {molecules[idx]['cid']:<12} {molecules[idx]['name'][:31]:<32} "
          f"{wt_best[i]:>5.1%} [{lo:>5.1%}, {hi:>5.1%}] {sel_freq[i]:>4.0%}")

# Pairwise rank stability
print("\n  Pairwise rank stability (fraction bootstrap A > B):")
n_sel = len(best_sel)
rank_stability = np.zeros((n_sel, n_sel))
for i in range(n_sel):
    for j in range(i + 1, n_sel):
        frac = np.mean(boot_w[:, i] > boot_w[:, j])
        rank_stability[i, j] = frac
        rank_stability[j, i] = 1 - frac
        if 0.3 < frac < 0.7:
            print(f"    ⚠ CID {molecules[best_sel[i]]['cid']} > "
                  f"CID {molecules[best_sel[j]]['cid']}: {frac:.0%} "
                  f"(rank order NOT robust)")

# Overlapping CIs
print("\n  Identifiability (overlapping 95% CIs):")
overlap_found = False
for i in range(n_sel):
    ci_i = (np.percentile(boot_w[:, i], 2.5), np.percentile(boot_w[:, i], 97.5))
    for j in range(i + 1, n_sel):
        ci_j = (np.percentile(boot_w[:, j], 2.5), np.percentile(boot_w[:, j], 97.5))
        if ci_i[0] < ci_j[1] and ci_j[0] < ci_i[1]:
            overlap_found = True
            print(f"    ⚠ CID {molecules[best_sel[i]]['cid']} and "
                  f"CID {molecules[best_sel[j]]['cid']} overlap")
if not overlap_found:
    print("    ✓ All component CIs are non-overlapping")

# ================================================================
# STEP 7: Peak-Resolved Residuals
# ================================================================
print("\n" + "=" * 72)
print("STEP 7: Peak-Resolved Residual Analysis")
print("=" * 72)

peaks, props = find_peaks(exp_norm, height=0.1, distance=5, prominence=0.05)
if len(peaks) > 0:
    pk_wn = exp_wn[peaks]
    pk_exp = exp_norm[peaks]
    pk_fit = recon_best[peaks]
    pk_res = pk_exp - pk_fit
    print(f"\n  Detected {len(peaks)} peaks (prominence > 0.05)")
    print(f"  Mean |residual| at peaks : {np.mean(np.abs(pk_res)):.4f}")
    print(f"  Max  |residual| at peaks : {np.max(np.abs(pk_res)):.4f}")
    print(f"\n  {'cm⁻¹':>8} {'Exp':>7} {'Fit':>7} {'Resid':>8} {'Rel Err':>8}")
    print("  " + "-" * 42)
    for w, e, f, r in zip(pk_wn, pk_exp, pk_fit, pk_res):
        rel = abs(r / e) * 100 if abs(e) > 0.01 else 0
        print(f"  {w:>8.1f} {e:>7.3f} {f:>7.3f} {r:>+8.3f} {rel:>7.1f}%")
else:
    print("  No peaks detected.")
    pk_wn, pk_res, pk_exp = np.array([]), np.array([]), np.array([])

# ================================================================
# STEP 8: Sensitivity to Scaling Factor
# ================================================================
print("\n" + "=" * 72)
print("STEP 8: Sensitivity to Scaling Factor")
print("=" * 72)

sens_results = []
print(f"\n  {'Scale':<7} {'Best k':<7} {'R²':>7} {'CV err':>8} {'Top CID':<12} {'Top Wt':>7}")
print("  " + "-" * 55)

for sf in SCALING_SENSITIVITY:
    shift = sf / SCALING_FACTOR
    dft_shifted = np.zeros_like(dft_matrix)
    for i in range(n_mol):
        dft_shifted[i] = np.interp(exp_wn, exp_wn * shift, dft_matrix[i],
                                   left=0, right=0)
    hist_s = forward_stepwise(dft_shifted, exp_wn, exp_norm, max_k=8)
    cv_s = [blocked_cv(dft_shifted, h[0], exp_wn, exp_norm) for h in hist_s]
    bk = np.argmin(cv_s) + 1
    sel_s = hist_s[bk - 1][0]
    dc_s, _, rss_s, _ = fit_lsq(dft_shifted[sel_s], exp_wn, exp_norm)
    r2_s = 1 - rss_s / ss_tot
    top_i = sel_s[np.argmax(dc_s)]
    top_w = dc_s.max() / dc_s.sum() if dc_s.sum() > 0 else 0
    sens_results.append({"scale": sf, "k": bk, "r2": r2_s, "cv": cv_s[bk-1],
                         "top_cid": molecules[top_i]["cid"], "top_w": top_w,
                         "sel": sel_s})
    print(f"  {sf:<7.2f} {bk:<7d} {r2_s:>7.4f} {cv_s[bk-1]:>8.4f} "
          f"{molecules[top_i]['cid']:<12} {top_w:>6.1%}")

top_cids = [r["top_cid"] for r in sens_results]
if len(set(top_cids)) == 1:
    print(f"\n  ✓ Top assignment stable across scales: CID {top_cids[0]}")
else:
    print(f"\n  ⚠ Top assignment varies: {top_cids}")

# ================================================================
# STEP 9: Sensitivity to Experimental Smoothing (Savitzky-Golay)
# ================================================================
print("\n" + "=" * 72)
print("STEP 9: Sensitivity to Experimental Smoothing (Savitzky-Golay)")
print("=" * 72)

# Load raw unsmoothed experimental spectrum
raw_df = pd.read_csv(EXP_RAW_FILE)
raw_wn = raw_df["Wavenumber"].values
raw_int = raw_df.iloc[:, 1].values  # second column (m/z 140 -ln(depl))
raw_mask = (raw_wn >= WAVENUMBER_RANGE[0]) & (raw_wn <= WAVENUMBER_RANGE[1])
raw_wn = raw_wn[raw_mask]
raw_int = raw_int[raw_mask]

# Reload DFT on raw wavenumber grid (may differ from smoothed exp grid)
_, dft_matrix_raw = load_dft_spectra(raw_wn, SCALING_FACTOR)
ss_tot_raw_base = None  # will compute per smoothing

sg_results = []
print(f"\n  {'SG(win,ord)':<14} {'Best k':<7} {'R²':>7} {'CV err':>8} "
      f"{'Top CID':<12} {'Top Wt':>7} {'Selected CIDs'}")
print("  " + "-" * 85)

for sg_win, sg_ord in SG_PARAMS:
    # Ensure window <= data length and is odd
    win = min(sg_win, len(raw_int))
    if win % 2 == 0:
        win -= 1
    if win <= sg_ord:
        continue

    # Apply Savitzky-Golay smoothing
    sg_smooth = savgol_filter(raw_int, window_length=win, polyorder=sg_ord)

    # Normalize (clip negative values from noise, then normalize by max)
    sg_smooth_clip = np.clip(sg_smooth, 0, None)
    sg_max = sg_smooth_clip.max()
    sg_norm = sg_smooth_clip / sg_max if sg_max > 0 else sg_smooth_clip

    ss_tot_sg = np.sum((sg_norm - sg_norm.mean())**2)

    # Forward stepwise on this smoothed version
    hist_sg = forward_stepwise(dft_matrix_raw, raw_wn, sg_norm, max_k=8)
    cv_sg = [blocked_cv(dft_matrix_raw, h[0], raw_wn, sg_norm) for h in hist_sg]
    bk_sg = np.argmin(cv_sg) + 1
    sel_sg = hist_sg[bk_sg - 1][0]
    dc_sg, _, rss_sg, _ = fit_lsq(dft_matrix_raw[sel_sg], raw_wn, sg_norm)
    r2_sg = 1 - rss_sg / ss_tot_sg
    top_i_sg = sel_sg[np.argmax(dc_sg)]
    top_w_sg = dc_sg.max() / dc_sg.sum() if dc_sg.sum() > 0 else 0
    wt_sg = dc_sg / dc_sg.sum() if dc_sg.sum() > 0 else dc_sg

    sel_cids = [molecules[s]["cid"] for s in sel_sg]
    sg_results.append({
        "window": sg_win, "polyorder": sg_ord, "k": bk_sg, "r2": r2_sg,
        "cv": cv_sg[bk_sg - 1], "top_cid": molecules[top_i_sg]["cid"],
        "top_w": top_w_sg, "sel": sel_sg, "weights": wt_sg,
        "sel_cids": sel_cids, "sg_norm": sg_norm
    })
    print(f"  ({sg_win:>2},{sg_ord})       {bk_sg:<7d} {r2_sg:>7.4f} {cv_sg[bk_sg-1]:>8.4f} "
          f"{molecules[top_i_sg]['cid']:<12} {top_w_sg:>6.1%} {sel_cids}")

# Also include the pre-smoothed spectrum (from the main pipeline) for comparison
print(f"\n  {'Presm.(2ch)':<14} {best_k:<7d} {r2_best:>7.4f} {'—':>8} "
      f"{molecules[best_sel[np.argmax(wt_best)]]['cid']:<12} {wt_best.max():>6.1%} "
      f"{[molecules[s]['cid'] for s in best_sel]}")

# Stability summary
sg_top_cids = [r["top_cid"] for r in sg_results]
if len(set(sg_top_cids)) == 1:
    print(f"\n  ✓ Top assignment stable across SG smoothings: CID {sg_top_cids[0]}")
else:
    print(f"\n  ⚠ Top assignment varies with smoothing: {sg_top_cids}")

# Check which CIDs appear consistently
all_sel_cids_sg = [cid for r in sg_results for cid in r["sel_cids"]]
cid_freq = Counter(all_sel_cids_sg)
n_sg_runs = len(sg_results)
print(f"\n  CID selection frequency across {n_sg_runs} smoothings:")
for cid, count in cid_freq.most_common():
    print(f"    CID {cid:<10} selected in {count}/{n_sg_runs} ({count/n_sg_runs:.0%})")

# --- Best SG smoothing: full fit report ---
best_sg_idx = np.argmin([r["cv"] for r in sg_results])
best_sg = sg_results[best_sg_idx]
print(f"\n  ★ Best SG smoothing (lowest CV error): "
      f"window={best_sg['window']}, polyorder={best_sg['polyorder']}")
print(f"    k={best_sg['k']}, R²={best_sg['r2']:.4f}, CV err={best_sg['cv']:.4f}")

# Full detailed fit with the best SG smoothing
sg_best_norm = best_sg["sg_norm"]
sg_best_sel = best_sg["sel"]
dc_sg_best, bc_sg_best, rss_sg_best, resid_sg_best = fit_lsq(
    dft_matrix_raw[sg_best_sel], raw_wn, sg_best_norm)
recon_sg_best = sg_best_norm - resid_sg_best
wt_sg_best = dc_sg_best / dc_sg_best.sum() if dc_sg_best.sum() > 0 else dc_sg_best

print(f"\n    {'#':<3} {'CID':<12} {'Name':<32} {'Wt':>6}")
print("    " + "-" * 56)
for i, idx in enumerate(sg_best_sel):
    print(f"    {i+1:<3} {molecules[idx]['cid']:<12} "
          f"{molecules[idx]['name'][:31]:<32} {wt_sg_best[i]:>5.1%}")

# Bootstrap on best SG smoothing
print(f"\n    Running bootstrap on best SG smoothing ({N_BOOTSTRAP} iterations)...")
boot_w_sg = run_bootstrap(dft_matrix_raw, sg_best_sel, raw_wn, sg_best_norm,
                          N_BOOTSTRAP, block_len)
sel_freq_sg = np.mean(boot_w_sg > 1e-4, axis=0)

print(f"\n    {'#':<3} {'CID':<12} {'Name':<32} {'Wt':>6} {'95% CI':>17} {'Sel%':>5}")
print("    " + "-" * 80)
for i, idx in enumerate(sg_best_sel):
    lo = np.percentile(boot_w_sg[:, i], 2.5)
    hi = np.percentile(boot_w_sg[:, i], 97.5)
    print(f"    {i+1:<3} {molecules[idx]['cid']:<12} "
          f"{molecules[idx]['name'][:31]:<32} {wt_sg_best[i]:>5.1%} "
          f"[{lo:>5.1%}, {hi:>5.1%}] {sel_freq_sg[i]:>4.0%}")

# ================================================================
# OUTPUT: Summary CSV
# ================================================================
rows = []
for i in range(n_mol):
    rows.append({
        "CID": molecules[i]["cid"],
        "Name": molecules[i]["name"],
        "Pearson_Derivative": round(pearson_sc[i], 5),
        "Cosine_Similarity": round(cosine_sc[i], 5),
        "NNLS_Spectral_Weight_Pct": round(wt_full[i] * 100, 3),
        "Pearson_Rank": int(np.where(pearson_rank == i)[0][0]) + 1,
        "Cosine_Rank": int(np.where(cosine_rank == i)[0][0]) + 1,
        "NNLS_Rank": int(np.where(nnls_rank == i)[0][0]) + 1,
        "In_Best_Model": "Yes" if i in best_sel else "No",
        "Cluster_ID": int(cluster_labels[i]),
    })

summary_df = pd.DataFrame(rows).sort_values("NNLS_Spectral_Weight_Pct", ascending=False)
summary_csv = os.path.join(OUTPUT_DIR, f"140_spectral_analysis_v3_summary_{RUN_TIMESTAMP_FILE}.csv")
# Write metadata header
with open(summary_csv, "w") as f:
    f.write(f"# Script: {SCRIPT_NAME}\n")
    f.write(f"# Run: {RUN_TIMESTAMP}\n")
    f.write(f"# Scaling factor: {SCALING_FACTOR}\n")
    f.write(f"# FWHM: {FWHM_CM} cm-1\n")
    f.write(f"# Wavenumber range: {WAVENUMBER_RANGE}\n")
    f.write(f"# Best k (blocked CV): {best_k}\n")
    f.write(f"# R2 (best model): {r2_best:.4f}\n")
summary_df.to_csv(summary_csv, mode="a", index=False)
print(f"\nSummary CSV: {summary_csv}")

# ================================================================
# OUTPUT: PDF
# ================================================================
output_pdf = os.path.join(OUTPUT_DIR, f"140_spectral_analysis_v3_{RUN_TIMESTAMP_FILE}.pdf")

with PdfPages(output_pdf) as pdf:

    # --- Page 1: Rankings ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    tn = min(15, n_mol)
    tp = pearson_rank[:tn]
    ax1.barh(range(tn), pearson_sc[tp], color="teal", edgecolor="none")
    ax1.set_yticks(range(tn))
    ax1.set_yticklabels([f"CID {molecules[i]['cid']}" for i in tp], fontsize=7)
    ax1.set_xlabel("Pearson (smoothed ∂/∂ν)")
    ax1.set_title("Pearson on First Derivative", fontweight="bold", fontsize=9)
    ax1.invert_yaxis()
    tc = cosine_rank[:tn]
    ax2.barh(range(tn), cosine_sc[tc], color="steelblue", edgecolor="none")
    ax2.set_yticks(range(tn))
    ax2.set_yticklabels([f"CID {molecules[i]['cid']}" for i in tc], fontsize=7)
    ax2.set_xlabel("Cosine Similarity")
    ax2.set_title("Cosine (raw spectra)", fontweight="bold", fontsize=9)
    ax2.invert_yaxis()
    plt.tight_layout()
    pdf.savefig(fig, dpi=100); plt.close(fig)

    # --- Page 2: Forward stepwise R², BIC, CV ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ks = range(1, len(sw_history) + 1)
    r2s = [1 - h[1] / ss_tot for h in sw_history]
    axes[0].plot(ks, r2s, "o-", color="darkgreen", lw=2, ms=5)
    axes[0].axvline(best_k, color="crimson", ls="--", lw=1, label=f"CV k={best_k}")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("R²")
    axes[0].set_title("Forward Stepwise R²"); axes[0].legend()
    axes[0].set_ylim(bottom=0)

    axes[1].plot(ks, bic_vals, "s-", color="crimson", lw=2, ms=5)
    axes[1].axvline(best_k_bic, color="gray", ls="--", label=f"BIC k={best_k_bic}")
    axes[1].set_xlabel("k"); axes[1].set_ylabel(f"BIC (n_eff={N_EFF})")
    axes[1].set_title("BIC (bandwidth-corrected)"); axes[1].legend()

    axes[2].plot(ks, cv_vals, "D-", color="darkorange", lw=2, ms=5)
    axes[2].axvline(best_k_cv, color="gray", ls="--", label=f"CV k={best_k_cv}")
    axes[2].set_xlabel("k"); axes[2].set_ylabel("CV Error (RSS)")
    axes[2].set_title(f"{N_CV_BLOCKS}-fold Blocked CV"); axes[2].legend()

    plt.tight_layout()
    pdf.savefig(fig, dpi=100); plt.close(fig)

    # --- Page 3: Best fit + residual ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7),
                                    gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(exp_wn, exp_norm, "k-", lw=1.2, label="Experimental m/z 140")
    ax1.plot(exp_wn, recon_best, "r-", lw=1.0, alpha=0.85,
             label=f"Fit (k={best_k}, R²={r2_best:.3f})")
    cmap = plt.colormaps["tab10"]
    for i, idx in enumerate(best_sel):
        contrib = dft_matrix[idx] * dc_best[i]
        ax1.fill_between(exp_wn, 0, contrib, alpha=0.2, color=cmap(i % 10),
                         label=f"CID {molecules[idx]['cid']} ({wt_best[i]:.0%})")
    ax1.set_xlim(WAVENUMBER_RANGE); ax1.set_ylabel("Intensity (norm.)")
    ax1.set_title(f"{best_k}-Component Decomposition "
                  "(spectral weights, NOT populations)", fontweight="bold", fontsize=9)
    ax1.legend(fontsize=6, loc="upper right", ncol=2)

    ax2.plot(exp_wn, resid_best, "k-", lw=0.8)
    ax2.axhline(0, color="gray", ls="--", lw=0.5)
    ax2.fill_between(exp_wn, 0, resid_best, alpha=0.25, color="gray")
    if len(pk_wn) > 0:
        ax2.scatter(pk_wn, pk_res, c="red", s=15, zorder=5, label="Peak residuals")
        ax2.legend(fontsize=7)
    ax2.set_xlim(WAVENUMBER_RANGE)
    ax2.set_xlabel("Wavenumber (cm$^{-1}$)"); ax2.set_ylabel("Residual")
    plt.tight_layout()
    pdf.savefig(fig, dpi=100); plt.close(fig)

    # --- Page 4: Bootstrap distributions ---
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot([boot_w[:, i] * 100 for i in range(n_sel)],
                    vert=True, patch_artist=True,
                    medianprops=dict(color="black", lw=1.5))
    for patch, i in zip(bp["boxes"], range(n_sel)):
        patch.set_facecolor(cmap(i % 10)); patch.set_alpha(0.5)
    ax.set_xticklabels([f"CID {molecules[idx]['cid']}" for idx in best_sel],
                       fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Spectral Weight (%)")
    ax.set_title(f"Block Bootstrap (n={N_BOOTSTRAP}, block={block_len}pt)",
                 fontweight="bold")
    # Overlay selection frequency
    for i in range(n_sel):
        ax.annotate(f"sel {sel_freq[i]:.0%}", (i + 1, np.percentile(boot_w[:, i], 97.5) * 100),
                    fontsize=6, ha="center", va="bottom", color="red")
    plt.tight_layout()
    pdf.savefig(fig, dpi=100); plt.close(fig)

    # --- Page 5: Sensitivity ---
    fig, ax = plt.subplots(figsize=(8, 5))
    scales = [r["scale"] for r in sens_results]
    r2s_s = [r["r2"] for r in sens_results]
    ax.plot(scales, r2s_s, "o-", color="purple", lw=2, ms=8)
    for r in sens_results:
        ax.annotate(f"k={r['k']}\nCID {r['top_cid']}", (r["scale"], r["r2"]),
                    textcoords="offset points", xytext=(0, 12), fontsize=7, ha="center")
    ax.set_xlabel("DFT Scaling Factor"); ax.set_ylabel("R² (best model)")
    ax.set_title("Sensitivity: R² vs Scaling Factor", fontweight="bold")
    plt.tight_layout()
    pdf.savefig(fig, dpi=100); plt.close(fig)

    # --- Page 6: Selected structures with individual spectra ---
    n_best = len(best_sel)
    # Up to 4 per page
    panels_per_page = 4
    n_struct_pages = int(np.ceil(n_best / panels_per_page))
    for sp in range(n_struct_pages):
        start_i = sp * panels_per_page
        end_i = min(start_i + panels_per_page, n_best)
        n_pan = end_i - start_i
        fig = plt.figure(figsize=(12, 3.2 * n_pan))
        gs = GridSpec(n_pan, 2, width_ratios=[3, 1], wspace=0.05, hspace=0.35)
        for pi, ci in enumerate(range(start_i, end_i)):
            idx = best_sel[ci]
            mol = molecules[idx]
            color = cmap(ci % 10)
            # Spectrum axis
            ax_sp = fig.add_subplot(gs[pi, 0])
            ax_sp.plot(exp_wn, exp_norm, color="0.5", lw=0.7, alpha=0.5, label="Exp")
            contrib = dft_matrix[idx] * dc_best[ci]
            ax_sp.fill_between(exp_wn, 0, contrib, alpha=0.3, color=color)
            ax_sp.plot(exp_wn, dft_matrix[idx] / dft_matrix[idx].max() if dft_matrix[idx].max() > 0 else dft_matrix[idx],
                       color=color, lw=1.0, label=f"DFT (norm.)")
            lo_ci = np.percentile(boot_w[:, ci], 2.5)
            hi_ci = np.percentile(boot_w[:, ci], 97.5)
            ax_sp.set_xlim(WAVENUMBER_RANGE)
            ax_sp.set_ylim(bottom=-0.05)
            ax_sp.text(0.01, 0.95,
                       f"CID {mol['cid']}\n{mol['name'][:40]}\n"
                       f"Wt: {wt_best[ci]:.1%} [{lo_ci:.1%}–{hi_ci:.1%}]  sel: {sel_freq[ci]:.0%}",
                       transform=ax_sp.transAxes, fontsize=7, va="top",
                       fontweight="bold", color=color)
            ax_sp.spines["top"].set_visible(False)
            ax_sp.spines["right"].set_visible(False)
            if pi == n_pan - 1:
                ax_sp.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=9)
            else:
                ax_sp.set_xticklabels([])
            ax_sp.set_ylabel("Int.", fontsize=8)
            # Image axis
            ax_im = fig.add_subplot(gs[pi, 1])
            ax_im.axis("off")
            img_path = os.path.join(IMG_DIR, f"CID_{mol['cid']}.png")
            if os.path.exists(img_path):
                try:
                    img = imread(img_path)
                    ax_im.imshow(img, aspect="equal")
                except Exception:
                    ax_im.text(0.5, 0.5, "No image", ha="center", va="center",
                               transform=ax_im.transAxes, fontsize=8)
            else:
                ax_im.text(0.5, 0.5, "No image", ha="center", va="center",
                           transform=ax_im.transAxes, fontsize=8)
        fig.suptitle(f"Selected Components (page {sp+1}/{n_struct_pages}) — "
                     f"spectral weights, NOT populations",
                     fontsize=10, fontweight="bold", y=0.99)
        pdf.savefig(fig, dpi=100); plt.close(fig)

    # --- Page: Smoothing Sensitivity (Step 9) ---
    if sg_results:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8),
                                 gridspec_kw={"height_ratios": [2, 1]})
        # Top: overlay all SG-smoothed spectra + the pre-smoothed one
        ax_sp = axes[0]
        ax_sp.plot(exp_wn, exp_norm, "k-", lw=1.5, label="Pre-smoothed (2ch)", zorder=10)
        cmap_sg = plt.colormaps["viridis"]
        for i, r in enumerate(sg_results):
            c = cmap_sg(i / max(len(sg_results) - 1, 1))
            ax_sp.plot(raw_wn, r["sg_norm"], "-", color=c, lw=0.8, alpha=0.7,
                       label=f"SG({r['window']},{r['polyorder']})")
        ax_sp.set_xlim(WAVENUMBER_RANGE)
        ax_sp.set_ylabel("Intensity (norm.)")
        ax_sp.set_title("Smoothing Sensitivity: SG-smoothed spectra from raw data",
                        fontweight="bold", fontsize=9)
        ax_sp.legend(fontsize=6, loc="upper right", ncol=2)

        # Bottom: R² and k vs smoothing
        ax_rk = axes[1]
        sg_labels = [f"({r['window']},{r['polyorder']})" for r in sg_results]
        r2_sg_vals = [r["r2"] for r in sg_results]
        k_sg_vals = [r["k"] for r in sg_results]
        x_pos = range(len(sg_results))
        ax_rk.bar(x_pos, r2_sg_vals, color="teal", alpha=0.6, label="R²")
        ax_rk.set_ylabel("R²", color="teal")
        ax_rk.set_ylim(0, 1)
        ax_rk.set_xticks(x_pos)
        ax_rk.set_xticklabels(sg_labels, fontsize=8)
        ax_rk.set_xlabel("SG (window, polyorder)")
        ax_rk2 = ax_rk.twinx()
        ax_rk2.plot(x_pos, k_sg_vals, "rs-", ms=7, lw=2, label="Best k")
        ax_rk2.set_ylabel("Best k", color="red")
        ax_rk2.set_ylim(0, max(k_sg_vals) + 2)
        ax_rk.legend(loc="upper left", fontsize=7)
        ax_rk2.legend(loc="upper right", fontsize=7)
        plt.tight_layout()
        pdf.savefig(fig, dpi=100); plt.close(fig)

    # --- Page: Best SG Fit (decomposition + residual) ---
    if sg_results:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7),
                                        gridspec_kw={"height_ratios": [3, 1]})
        ss_tot_sg_best = np.sum((sg_best_norm - sg_best_norm.mean())**2)
        r2_sg_b = 1 - rss_sg_best / ss_tot_sg_best
        ax1.plot(raw_wn, sg_best_norm, "k-", lw=1.2,
                 label=f"Raw + SG({best_sg['window']},{best_sg['polyorder']})")
        ax1.plot(raw_wn, recon_sg_best, "r-", lw=1.0, alpha=0.85,
                 label=f"Fit (k={best_sg['k']}, R²={r2_sg_b:.3f})")
        cmap_b = plt.colormaps["tab10"]
        for i, idx in enumerate(sg_best_sel):
            contrib = dft_matrix_raw[idx] * dc_sg_best[i]
            ax1.fill_between(raw_wn, 0, contrib, alpha=0.2, color=cmap_b(i % 10),
                             label=f"CID {molecules[idx]['cid']} ({wt_sg_best[i]:.0%})")
        ax1.set_xlim(WAVENUMBER_RANGE); ax1.set_ylabel("Intensity (norm.)")
        ax1.set_title(f"Best SG Smoothing Decomposition — SG({best_sg['window']},"
                      f"{best_sg['polyorder']}), k={best_sg['k']}\n"
                      f"(spectral weights, NOT populations)",
                      fontweight="bold", fontsize=9)
        ax1.legend(fontsize=6, loc="upper right", ncol=2)

        ax2.plot(raw_wn, resid_sg_best, "k-", lw=0.8)
        ax2.axhline(0, color="gray", ls="--", lw=0.5)
        ax2.fill_between(raw_wn, 0, resid_sg_best, alpha=0.25, color="gray")
        ax2.set_xlim(WAVENUMBER_RANGE)
        ax2.set_xlabel("Wavenumber (cm$^{-1}$)"); ax2.set_ylabel("Residual")
        plt.tight_layout()
        pdf.savefig(fig, dpi=100); plt.close(fig)

        # --- Page: Best SG Bootstrap boxplots ---
        n_sg_sel = len(sg_best_sel)
        fig, ax = plt.subplots(figsize=(10, 5))
        bp = ax.boxplot([boot_w_sg[:, i] * 100 for i in range(n_sg_sel)],
                        vert=True, patch_artist=True,
                        medianprops=dict(color="black", lw=1.5))
        for patch, i in zip(bp["boxes"], range(n_sg_sel)):
            patch.set_facecolor(cmap_b(i % 10)); patch.set_alpha(0.5)
        ax.set_xticklabels([f"CID {molecules[idx]['cid']}" for idx in sg_best_sel],
                           fontsize=7, rotation=45, ha="right")
        ax.set_ylabel("Spectral Weight (%)")
        ax.set_title(f"Best SG({best_sg['window']},{best_sg['polyorder']}) — "
                     f"Block Bootstrap (n={N_BOOTSTRAP})",
                     fontweight="bold")
        for i in range(n_sg_sel):
            ax.annotate(f"sel {sel_freq_sg[i]:.0%}",
                        (i + 1, np.percentile(boot_w_sg[:, i], 97.5) * 100),
                        fontsize=6, ha="center", va="bottom", color="red")
        plt.tight_layout()
        pdf.savefig(fig, dpi=100); plt.close(fig)

    # --- Page 7+: Gram matrix heatmap ---
    fig, ax = plt.subplots(figsize=(10, 9))
    gram_diag = dft_normed @ dft_normed.T
    im = ax.imshow(gram_diag, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n_mol)); ax.set_yticks(range(n_mol))
    lbl = [molecules[i]["cid"] for i in range(n_mol)]
    ax.set_xticklabels(lbl, fontsize=4, rotation=90)
    ax.set_yticklabels(lbl, fontsize=4)
    ax.set_title("DFT-DFT Cosine Similarity (Gram Matrix)", fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    pdf.savefig(fig, dpi=100); plt.close(fig)

print(f"\nPDF: {output_pdf}")
print("\n" + "=" * 72)
print("  CAVEATS (for methods section)")
print("=" * 72)
print(f"""
  1. Spectral weights ≠ populations. The IR-UV ion-dip signal samples
     the UV-ionizable subpopulation, weighted by σ_UV(ν_UV) × ϕ_ion.
     Absence claims cannot be made for UV-silent candidates.

  2. Collinearity (cond={cond_num:.1e}, {n_90} pairs cos>0.90) limits
     identifiability. Bootstrap CIs and selection frequencies above
     indicate which assignments are robust vs. collinearity-driven.

  3. DFT harmonic frequencies (scaled {SCALING_FACTOR}) may not match
     IRMPD/ion-dip positions exactly (anharmonicity, temperature).

  4. Model selection used {N_CV_BLOCKS}-fold blocked CV (k={best_k})
     with n_eff={N_EFF} bandwidth-corrected BIC as cross-check
     (BIC k={best_k_bic}).

  5. No upstream σ weighting (experimental CSV lacks per-point
     uncertainties). If available, feed σ into weighted lsq_linear
     for χ²/dof diagnostics.
""")
print("Done!")
