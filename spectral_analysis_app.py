import io
import re
import datetime
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import lsq_linear
from scipy.signal import find_peaks, savgol_filter
from scipy.cluster.hierarchy import linkage, fcluster
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ================================================================
# DEFAULTS
# ================================================================
DEFAULT_CONFIG = {
    "wn_min": 400.0,
    "wn_max": 1600.0,
    "fwhm_cm": 10.0,
    "scaling_factor": 0.95,
    "scaling_sensitivity": [0.94, 0.95, 0.96, 0.97],
    "n_bootstrap": 1000,
    "max_k_exhaustive": 5,
    "max_k_forward": 12,
    "n_cv_blocks": 10,
    "poly_order": 1,
    "nonzero_threshold": 0.0,
    "peak_height": 0.1,
    "peak_distance": 5,
    "peak_prominence": 0.05,
    "sg_params": [(5, 2), (9, 2), (15, 3), (21, 3), (31, 3), (41, 4)],
    "dft_regex": r"CID_(\d+)_(.+)_scaled_([0-9.]+)\.csv",
    "dft_col_wn": "Wavenumber",
    "dft_col_int": "Total",
}


# ================================================================
# DATA LOADING
# ================================================================
@st.cache_data(show_spinner=False)
def load_experimental_from_upload(uploaded_file, wn_col, int_col, wn_range):
    df = pd.read_csv(uploaded_file)
    if wn_col not in df.columns:
        raise ValueError(f"Wavenumber column '{wn_col}' not found. Columns: {list(df.columns)}")
    if int_col not in df.columns:
        raise ValueError(f"Intensity column '{int_col}' not found. Columns: {list(df.columns)}")
    wn = df[wn_col].values.astype(float)
    intens = df[int_col].values.astype(float)
    mask = (wn >= wn_range[0]) & (wn <= wn_range[1])
    wn, intens = wn[mask], intens[mask]
    if len(wn) < 3:
        raise ValueError(f"Only {len(wn)} points after wavenumber range mask.")
    imax = intens.max()
    norm = intens / imax if imax > 0 else intens
    return wn, norm, intens, imax


@st.cache_data(show_spinner=False)
def load_dft_from_uploads(uploaded_files, exp_wn, scale_factor, regex_pattern, wn_col, int_col):
    molecules = []
    dft_list = []
    pattern = re.compile(regex_pattern)
    for f in uploaded_files:
        name = f.name
        m = pattern.match(name)
        if not m:
            continue
        cid, mol_name = m.group(1), m.group(2).replace("_", " ")
        # Read CSV, skip possible comment lines
        df = pd.read_csv(f, comment="#")
        # If the file has no header, the first row becomes the header by default.
        # Try to infer numeric columns.
        if df[df.columns[0]].dtype.kind not in "iuf":
            # No header: re-read with header=None
            f.seek(0)
            df = pd.read_csv(f, comment="#", header=None)
            df.columns = ["Wavenumber", "Fundamentals", "Overtones", "Combinations", "Total"]

        if wn_col not in df.columns:
            # fallback to first numeric column
            wn_col_actual = df.columns[0]
        else:
            wn_col_actual = wn_col
        if int_col not in df.columns:
            int_col_actual = df.columns[-1]
        else:
            int_col_actual = int_col

        dft_wn = df[wn_col_actual].values.astype(float)
        dft_int = df[int_col_actual].values.astype(float)
        dft_interp = np.interp(exp_wn, dft_wn, dft_int, left=0.0, right=0.0)
        molecules.append({"cid": cid, "name": mol_name, "file": name})
        dft_list.append(dft_interp)
    if not molecules:
        raise ValueError(
            "No DFT files matched the expected naming pattern. "
            f"Pattern used: {regex_pattern}. Example: CID_12345_molecule_name_scaled_0.95.csv"
        )
    return molecules, np.array(dft_list)


# ================================================================
# ANALYSIS FUNCTIONS
# ================================================================
def build_A(dft_sub, exp_wn, poly_order=1):
    n_wn = len(exp_wn)
    wn_c = 2 * (exp_wn - exp_wn.min()) / (exp_wn.max() - exp_wn.min()) - 1
    base_cols = [wn_c ** p for p in range(poly_order + 1)]
    B = np.column_stack(base_cols)
    A = np.hstack([dft_sub.T, B])
    return A, dft_sub.shape[0], B.shape[1]


def fit_lsq(dft_sub, exp_wn, exp_norm, poly_order=1):
    A, nd, nb = build_A(dft_sub, exp_wn, poly_order)
    lb = np.concatenate([np.zeros(nd), np.full(nb, -np.inf)])
    ub = np.full(nd + nb, np.inf)
    res = lsq_linear(A, exp_norm, bounds=(lb, ub), method="bvls")
    dc, bc = res.x[:nd], res.x[nd:]
    resid = exp_norm - A @ res.x
    rss = float(np.sum(resid ** 2))
    return dc, bc, rss, resid


def felix_smooth(spectrum, wn_axis, bw_frac=0.0053):
    n = len(spectrum)
    smoothed = np.zeros(n)
    for i in range(n):
        fwhm_i = bw_frac * wn_axis[i]
        sigma_i = fwhm_i / (2 * np.sqrt(2 * np.log(2)))
        dists = np.abs(wn_axis - wn_axis[i])
        kernel = np.exp(-0.5 * (dists / sigma_i) ** 2)
        kernel /= kernel.sum()
        smoothed[i] = np.dot(kernel, spectrum)
    return smoothed


def pearson_deriv_scores(exp_norm, dft_matrix, exp_wn, bw_frac=0.0053):
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
            scores.append(float(np.dot(ed_c, dd_c) / (ed_n * dd_n)))
        else:
            scores.append(0.0)
    return np.array(scores)


def cosine_scores(exp_norm, dft_matrix):
    ne = np.linalg.norm(exp_norm)
    return np.array([
        float(np.dot(exp_norm, d) / (ne * np.linalg.norm(d))) if np.linalg.norm(d) > 0 else 0.0
        for d in dft_matrix
    ])


def forward_stepwise(dft_matrix, exp_wn, exp_norm, max_k, poly_order=1):
    available = set(range(dft_matrix.shape[0]))
    selected = []
    history = []
    for _ in range(max_k):
        best_rss, best_i = np.inf, None
        for cand in available:
            trial = selected + [cand]
            _, _, rss, _ = fit_lsq(dft_matrix[trial], exp_wn, exp_norm, poly_order)
            if rss < best_rss:
                best_rss, best_i = rss, cand
        if best_i is None:
            break
        selected.append(best_i)
        available.remove(best_i)
        history.append((list(selected), float(best_rss)))
    return history


def exhaustive_search(dft_matrix, exp_wn, exp_norm, k, poly_order=1):
    best_rss, best_sub = np.inf, None
    for sub in combinations(range(dft_matrix.shape[0]), k):
        _, _, rss, _ = fit_lsq(dft_matrix[list(sub)], exp_wn, exp_norm, poly_order)
        if rss < best_rss:
            best_rss, best_sub = rss, list(sub)
    return best_sub, float(best_rss)


def bic_neff(rss, k, n_eff):
    return n_eff * np.log(rss / n_eff) + k * np.log(n_eff)


def blocked_cv(dft_matrix, selected, exp_wn, exp_norm, n_blocks, poly_order=1):
    n = len(exp_norm)
    edges = np.linspace(0, n, n_blocks + 1, dtype=int)
    cv_err = 0.0
    sub = dft_matrix[selected]
    for b in range(n_blocks):
        test_mask = np.zeros(n, dtype=bool)
        test_mask[edges[b]:edges[b + 1]] = True
        train_mask = ~test_mask
        A_tr, nd, nb = build_A(sub[:, train_mask], exp_wn[train_mask], poly_order)
        lb = np.concatenate([np.zeros(nd), np.full(nb, -np.inf)])
        ub = np.full(nd + nb, np.inf)
        res = lsq_linear(A_tr, exp_norm[train_mask], bounds=(lb, ub), method="bvls")
        A_te, _, _ = build_A(sub[:, test_mask], exp_wn[test_mask], poly_order)
        pred = A_te @ res.x
        cv_err += float(np.sum((exp_norm[test_mask] - pred) ** 2))
    return cv_err


def block_bootstrap_indices(n, block_len, rng):
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, max(n - block_len + 1, 1), size=n_blocks)
    idx = np.concatenate([np.arange(s, min(s + block_len, n)) for s in starts])
    return idx[:n]


def run_bootstrap(dft_matrix, selected, exp_wn, exp_norm, n_boot, block_len, poly_order=1, seed=42):
    rng = np.random.default_rng(seed)
    n_sel = len(selected)
    all_weights = np.zeros((n_boot, n_sel))
    sub = dft_matrix[selected]
    for b in range(n_boot):
        idx = block_bootstrap_indices(len(exp_norm), block_len, rng)
        dc, _, _, _ = fit_lsq(sub[:, idx], exp_wn[idx], exp_norm[idx], poly_order)
        s = dc.sum()
        all_weights[b] = dc / s if s > 0 else dc
    return all_weights


# ================================================================
# PLOTTING HELPERS
# ================================================================
def make_ranking_plot(molecules, scores, title, key="score"):
    top_n = min(15, len(molecules))
    idx = np.argsort(scores)[::-1][:top_n]
    labels = [f"CID {molecules[i]['cid']}" for i in idx]
    vals = scores[idx]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=vals, y=labels, orientation="h", marker_color="teal"))
    fig.update_layout(
        title=title,
        xaxis_title=key,
        yaxis=dict(autorange="reversed"),
        height=500,
        margin=dict(l=120),
    )
    return fig


def make_fit_plot(exp_wn, exp_norm, recon, best_sel, dft_matrix, molecules, best_k, r2_best):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.08,
        subplot_titles=("Best Fit", "Residual"),
    )
    fig.add_trace(go.Scatter(x=exp_wn, y=exp_norm, mode="lines", name="Experimental", line=dict(color="black")), row=1, col=1)
    fig.add_trace(go.Scatter(x=exp_wn, y=recon, mode="lines", name=f"Fit (k={best_k}, R²={r2_best:.3f})", line=dict(color="red")), row=1, col=1)
    colors = px.colors.qualitative.Plotly
    for i, idx in enumerate(best_sel):
        contribution = dft_matrix[idx] * fit_lsq(dft_matrix[best_sel], exp_wn, exp_norm, poly_order=1)[0][i]
        fig.add_trace(go.Scatter(
            x=exp_wn, y=contribution,
            mode="lines",
            name=f"CID {molecules[idx]['cid']} ({molecules[idx]['name'][:20]})",
            line=dict(color=colors[i % len(colors)]),
            opacity=0.7,
        ), row=1, col=1)
    residual = exp_norm - recon
    fig.add_trace(go.Scatter(x=exp_wn, y=residual, mode="lines", name="Residual", line=dict(color="gray")), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)
    fig.update_xaxes(title_text="Wavenumber (cm⁻¹)", row=2, col=1)
    fig.update_yaxes(title_text="Intensity (norm.)", row=1, col=1)
    fig.update_yaxes(title_text="Residual", row=2, col=1)
    fig.update_layout(height=700, showlegend=True, legend=dict(font_size=9))
    return fig


def make_stepwise_plot(ks, r2s, bic_vals, cv_vals, best_k_bic, best_k_cv, n_eff, n_cv_blocks):
    fig = make_subplots(rows=1, cols=3, subplot_titles=("R²", "BIC", "Blocked CV"))
    fig.add_trace(go.Scatter(x=list(ks), y=r2s, mode="lines+markers", name="R²"), row=1, col=1)
    fig.add_vline(x=best_k_cv, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_trace(go.Scatter(x=list(ks), y=bic_vals, mode="lines+markers", name="BIC", marker_color="crimson"), row=1, col=2)
    fig.add_vline(x=best_k_bic, line_dash="dash", line_color="gray", row=1, col=2)
    fig.add_trace(go.Scatter(x=list(ks), y=cv_vals, mode="lines+markers", name="CV", marker_color="darkorange"), row=1, col=3)
    fig.add_vline(x=best_k_cv, line_dash="dash", line_color="gray", row=1, col=3)
    fig.update_xaxes(title_text="k", row=1, col=1)
    fig.update_xaxes(title_text="k", row=1, col=2)
    fig.update_xaxes(title_text="k", row=1, col=3)
    fig.update_yaxes(title_text="R²", row=1, col=1)
    fig.update_yaxes(title_text=f"BIC (n_eff={n_eff})", row=1, col=2)
    fig.update_yaxes(title_text=f"{n_cv_blocks}-fold CV RSS", row=1, col=3)
    fig.update_layout(height=400, showlegend=False)
    return fig


def make_bootstrap_plot(best_sel, boot_w, molecules):
    fig = go.Figure()
    positions = list(range(len(best_sel)))
    for i, idx in enumerate(best_sel):
        fig.add_trace(go.Box(
            y=boot_w[:, i],
            name=f"CID {molecules[idx]['cid']}",
            boxpoints="outliers",
            marker_color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)],
        ))
    fig.update_layout(
        title="Bootstrap Spectral Weight Distributions",
        yaxis_title="Spectral weight",
        xaxis_title="Component",
        height=500,
    )
    return fig


def make_sensitivity_plot(sens_results):
    scales = [r["scale"] for r in sens_results]
    r2s = [r["r2"] for r in sens_results]
    top_cids = [r["top_cid"] for r in sens_results]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=scales, y=r2s, mode="lines+markers", name="R²"))
    for x, y, cid in zip(scales, r2s, top_cids):
        fig.add_annotation(x=x, y=y, text=f"k={next(r['k'] for r in sens_results if r['scale']==x)}<br>top {cid}", showarrow=False, yshift=15, font_size=9)
    fig.update_layout(title="Sensitivity to DFT Scaling Factor", xaxis_title="Scaling factor", yaxis_title="R²", height=500)
    return fig


def make_gram_matrix_plot(molecules, dft_matrix):
    norms = np.linalg.norm(dft_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    gram = (dft_matrix / norms) @ (dft_matrix / norms).T
    labels = [m["cid"] for m in molecules]
    fig = go.Figure(data=go.Heatmap(z=gram, x=labels, y=labels, colorscale="RdYlBu_r", zmin=0, zmax=1))
    fig.update_layout(title="DFT-DFT Cosine Similarity", height=700, xaxis_tickangle=-90)
    return fig


# ================================================================
# PDF REPORT
# ================================================================
def build_pdf_report(exp_wn, exp_norm, molecules, dft_matrix, best_sel, best_k, r2_best, recon_best,
                     pearson_sc, cosine_sc, sw_history, bic_vals, cv_vals, best_k_bic, best_k_cv,
                     boot_w, sens_results, n_eff, n_cv_blocks, config):
    pdf_buffer = io.BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        # Page 1: rankings
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        tn = min(15, len(molecules))
        tp = np.argsort(pearson_sc)[::-1][:tn]
        ax1.barh(range(tn), pearson_sc[tp], color="teal")
        ax1.set_yticks(range(tn))
        ax1.set_yticklabels([f"CID {molecules[i]['cid']}" for i in tp], fontsize=7)
        ax1.set_xlabel("Pearson (smoothed ∂/∂ν)")
        ax1.set_title("Pearson on First Derivative")
        ax1.invert_yaxis()
        tc = np.argsort(cosine_sc)[::-1][:tn]
        ax2.barh(range(tn), cosine_sc[tc], color="steelblue")
        ax2.set_yticks(range(tn))
        ax2.set_yticklabels([f"CID {molecules[i]['cid']}" for i in tc], fontsize=7)
        ax2.set_xlabel("Cosine Similarity")
        ax2.set_title("Cosine (raw spectra)")
        ax2.invert_yaxis()
        plt.tight_layout()
        pdf.savefig(fig, dpi=100)
        plt.close(fig)

        # Page 2: stepwise
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        ks = list(range(1, len(sw_history) + 1))
        ss_tot = np.sum((exp_norm - exp_norm.mean()) ** 2)
        r2s = [1 - h[1] / ss_tot for h in sw_history]
        axes[0].plot(ks, r2s, "o-", color="darkgreen")
        axes[0].axvline(best_k_cv, color="crimson", ls="--", label=f"CV k={best_k_cv}")
        axes[0].set_xlabel("k")
        axes[0].set_ylabel("R²")
        axes[0].set_title("Forward Stepwise R²")
        axes[0].legend()
        axes[1].plot(ks, bic_vals, "s-", color="crimson")
        axes[1].axvline(best_k_bic, color="gray", ls="--", label=f"BIC k={best_k_bic}")
        axes[1].set_xlabel("k")
        axes[1].set_ylabel(f"BIC (n_eff={n_eff})")
        axes[1].set_title("BIC")
        axes[1].legend()
        axes[2].plot(ks, cv_vals, "D-", color="darkorange")
        axes[2].axvline(best_k_cv, color="gray", ls="--", label=f"CV k={best_k_cv}")
        axes[2].set_xlabel("k")
        axes[2].set_ylabel("CV Error")
        axes[2].set_title(f"{n_cv_blocks}-fold Blocked CV")
        axes[2].legend()
        plt.tight_layout()
        pdf.savefig(fig, dpi=100)
        plt.close(fig)

        # Page 3: best fit + residual
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [3, 1]})
        ax1.plot(exp_wn, exp_norm, "k-", lw=1.2, label="Experimental")
        ax1.plot(exp_wn, recon_best, "r-", lw=1.0, alpha=0.85, label=f"Fit (k={best_k}, R²={r2_best:.3f})")
        colors = plt.colormaps["tab10"]
        dc_best, _, _, _ = fit_lsq(dft_matrix[best_sel], exp_wn, exp_norm, poly_order=config["poly_order"])
        for i, idx in enumerate(best_sel):
            contribution = dft_matrix[idx] * dc_best[i]
            ax1.fill_between(exp_wn, 0, contribution, alpha=0.2, color=colors(i % 10),
                             label=f"CID {molecules[idx]['cid']}")
        ax1.set_xlim(config["wn_min"], config["wn_max"])
        ax1.set_ylabel("Intensity (norm.)")
        ax1.set_title("Best Fit Decomposition")
        ax1.legend(fontsize=6, loc="upper right", ncol=2)
        residual = exp_norm - recon_best
        ax2.plot(exp_wn, residual, "k-", lw=0.8)
        ax2.axhline(0, color="gray", ls="--", lw=0.5)
        ax2.fill_between(exp_wn, 0, residual, alpha=0.3, color="gray")
        ax2.set_xlim(config["wn_min"], config["wn_max"])
        ax2.set_xlabel("Wavenumber (cm⁻¹)")
        ax2.set_ylabel("Residual")
        plt.tight_layout()
        pdf.savefig(fig, dpi=100)
        plt.close(fig)

        # Page 4: bootstrap
        fig, ax = plt.subplots(figsize=(10, 5))
        bp = ax.boxplot([boot_w[:, i] * 100 for i in range(len(best_sel))], vert=True, patch_artist=True)
        for patch, i in zip(bp["boxes"], range(len(best_sel))):
            patch.set_facecolor(plt.colormaps["tab10"](i % 10))
            patch.set_alpha(0.5)
        ax.set_xticklabels([f"CID {molecules[idx]['cid']}" for idx in best_sel], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Spectral Weight (%)")
        ax.set_title(f"Bootstrap Distributions (n={len(boot_w)})")
        plt.tight_layout()
        pdf.savefig(fig, dpi=100)
        plt.close(fig)

        # Page 5: sensitivity
        fig, ax = plt.subplots(figsize=(10, 5))
        scales = [r["scale"] for r in sens_results]
        r2s_sf = [r["r2"] for r in sens_results]
        ax.plot(scales, r2s_sf, "o-", color="purple")
        ax.set_xlabel("DFT Scaling Factor")
        ax.set_ylabel("R²")
        ax.set_title("Sensitivity to Scaling Factor")
        for r in sens_results:
            ax.annotate(f"k={r['k']}", (r["scale"], r["r2"]), textcoords="offset points", xytext=(0, 10), fontsize=8, ha="center")
        plt.tight_layout()
        pdf.savefig(fig, dpi=100)
        plt.close(fig)

    pdf_buffer.seek(0)
    return pdf_buffer


# ================================================================
# STREAMLIT UI
# ================================================================
def main():
    st.set_page_config(page_title="Spectral Analysis", layout="wide")
    st.title("Spectral Analysis Pipeline")
    st.markdown("Upload an experimental spectrum and DFT candidate spectra, tune the criteria, and run the full analysis pipeline.")

    # ---- Sidebar: configuration ----
    st.sidebar.header("Upload Data")
    exp_file = st.sidebar.file_uploader("Experimental CSV", type="csv")
    dft_files = st.sidebar.file_uploader("DFT spectra CSVs (multiple)", type="csv", accept_multiple_files=True)

    st.sidebar.header("Analysis Settings")
    wn_min = st.sidebar.number_input("Wavenumber min (cm⁻¹)", value=DEFAULT_CONFIG["wn_min"], step=10.0)
    wn_max = st.sidebar.number_input("Wavenumber max (cm⁻¹)", value=DEFAULT_CONFIG["wn_max"], step=10.0)
    fwhm_cm = st.sidebar.number_input("FWHM (cm⁻¹)", value=DEFAULT_CONFIG["fwhm_cm"], step=1.0, min_value=0.1)
    scaling_factor = st.sidebar.number_input("DFT scaling factor", value=DEFAULT_CONFIG["scaling_factor"], step=0.01, format="%.2f")
    scaling_sensitivity = st.sidebar.text_input("Scaling sensitivity (comma-separated)", value=", ".join(map(str, DEFAULT_CONFIG["scaling_sensitivity"])))
    n_bootstrap = st.sidebar.number_input("Bootstrap iterations", value=DEFAULT_CONFIG["n_bootstrap"], step=100, min_value=100)
    max_k_forward = st.sidebar.number_input("Max forward stepwise k", value=DEFAULT_CONFIG["max_k_forward"], step=1, min_value=1)
    max_k_exhaustive = st.sidebar.number_input("Max k for exhaustive search", value=DEFAULT_CONFIG["max_k_exhaustive"], step=1, min_value=1)
    n_cv_blocks = st.sidebar.number_input("CV blocks", value=DEFAULT_CONFIG["n_cv_blocks"], step=1, min_value=2)
    poly_order = st.sidebar.selectbox("Baseline polynomial order", [0, 1, 2], index=1)

    st.sidebar.header("Peak Detection")
    peak_height = st.sidebar.number_input("Peak height", value=DEFAULT_CONFIG["peak_height"], step=0.05)
    peak_distance = st.sidebar.number_input("Peak distance (points)", value=DEFAULT_CONFIG["peak_distance"], step=1)
    peak_prominence = st.sidebar.number_input("Peak prominence", value=DEFAULT_CONFIG["peak_prominence"], step=0.01)

    st.sidebar.header("File Naming")
    dft_regex = st.sidebar.text_input("DFT filename regex", value=DEFAULT_CONFIG["dft_regex"])
    dft_wn_col = st.sidebar.text_input("DFT wavenumber column", value=DEFAULT_CONFIG["dft_col_wn"])
    dft_int_col = st.sidebar.text_input("DFT intensity column", value=DEFAULT_CONFIG["dft_col_int"])

    run = st.sidebar.button("Run Analysis", type="primary")

    config = {
        "wn_min": wn_min,
        "wn_max": wn_max,
        "fwhm_cm": fwhm_cm,
        "scaling_factor": scaling_factor,
        "scaling_sensitivity": [float(x.strip()) for x in scaling_sensitivity.split(",") if x.strip()],
        "n_bootstrap": int(n_bootstrap),
        "max_k_forward": int(max_k_forward),
        "max_k_exhaustive": int(max_k_exhaustive),
        "n_cv_blocks": int(n_cv_blocks),
        "poly_order": int(poly_order),
        "peak_height": peak_height,
        "peak_distance": int(peak_distance),
        "peak_prominence": peak_prominence,
        "dft_regex": dft_regex,
        "dft_wn_col": dft_wn_col,
        "dft_int_col": dft_int_col,
    }

    # ---- Main: load preview ----
    if exp_file is not None:
        try:
            df_preview = pd.read_csv(exp_file, nrows=3)
            exp_file.seek(0)
            st.subheader("Experimental file preview")
            st.write(df_preview)
            exp_wn_col = st.selectbox("Experimental wavenumber column", df_preview.columns.tolist())
            exp_int_col = st.selectbox("Experimental intensity column", df_preview.columns.tolist(), index=1 if len(df_preview.columns) > 1 else 0)
        except Exception as e:
            st.error(f"Error reading experimental file: {e}")
            return
    else:
        st.info("Upload an experimental CSV to begin.")
        return

    if not dft_files:
        st.info("Upload one or more DFT spectra CSVs.")
        return

    st.markdown(f"**{len(dft_files)}** DFT file(s) uploaded.")

    if not run:
        st.info("Click **Run Analysis** in the sidebar when ready.")
        return

    # ---- Run analysis ----
    progress = st.progress(0, text="Loading data...")
    try:
        wn_range = (config["wn_min"], config["wn_max"])
        exp_wn, exp_norm, exp_int, exp_max = load_experimental_from_upload(
            exp_file, exp_wn_col, exp_int_col, wn_range
        )
        n_eff = max(1, int((config["wn_max"] - config["wn_min"]) / config["fwhm_cm"]))
        molecules, dft_matrix = load_dft_from_uploads(
            dft_files, exp_wn, config["scaling_factor"], config["dft_regex"],
            config["dft_wn_col"], config["dft_int_col"]
        )
        n_mol = len(molecules)
        n_wn = len(exp_wn)
        wn_step = exp_wn[1] - exp_wn[0] if n_wn > 1 else 1.0
        block_len = max(int(3 * config["fwhm_cm"] / wn_step), 5)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    progress.progress(0.15, text="Computing diagnostics...")
    ss_tot = float(np.sum((exp_norm - exp_norm.mean()) ** 2))

    # Collinearity
    cond_num = float(np.linalg.cond(dft_matrix.T))
    norms = np.linalg.norm(dft_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    dft_normed = dft_matrix / norms
    gram = dft_normed @ dft_normed.T
    np.fill_diagonal(gram, 0)
    max_cos = float(gram.max()) if gram.size > 0 else 0.0
    i_mc, j_mc = (0, 0)
    if gram.size > 0:
        i_mc, j_mc = np.unravel_index(gram.argmax(), gram.shape)
    n_95 = int((gram > 0.95).sum() // 2)
    n_90 = int((gram > 0.90).sum() // 2)

    dist = 1 - np.abs(dft_normed @ dft_normed.T)
    condensed = dist[np.triu_indices(n_mol, k=1)]
    Z = linkage(condensed, method="average")
    cluster_labels = fcluster(Z, t=0.10, criterion="distance")

    progress.progress(0.25, text="Ranking candidates...")
    pearson_sc = pearson_deriv_scores(exp_norm, dft_matrix, exp_wn)
    cosine_sc = cosine_scores(exp_norm, dft_matrix)
    pearson_rank = np.argsort(pearson_sc)[::-1]
    cosine_rank = np.argsort(cosine_sc)[::-1]

    progress.progress(0.40, text="Full NNLS...")
    dc_full, bc_full, rss_full, resid_full = fit_lsq(dft_matrix, exp_wn, exp_norm, config["poly_order"])
    r2_full = 1 - rss_full / ss_tot if ss_tot > 0 else 0.0
    wt_full = dc_full / dc_full.sum() if dc_full.sum() > 0 else dc_full
    nnls_rank = np.argsort(wt_full)[::-1]

    progress.progress(0.55, text="Forward stepwise selection...")
    sw_history = forward_stepwise(dft_matrix, exp_wn, exp_norm, config["max_k_forward"], config["poly_order"])
    if not sw_history:
        st.error("Forward stepwise selection produced no models.")
        return

    bic_vals = []
    cv_vals = []
    for step, (sel, rss) in enumerate(sw_history):
        k = step + 1
        bic_vals.append(bic_neff(rss, k, n_eff))
        cv_vals.append(blocked_cv(dft_matrix, sel, exp_wn, exp_norm, config["n_cv_blocks"], config["poly_order"]))
    bic_vals = [float(v) for v in bic_vals]
    cv_vals = [float(v) for v in cv_vals]
    best_k_bic = int(np.argmin(bic_vals) + 1)
    best_k_cv = int(np.argmin(cv_vals) + 1)
    best_k = best_k_cv
    best_sel_sw = sw_history[best_k - 1][0]

    # Exhaustive search
    if best_k <= config["max_k_exhaustive"]:
        progress.progress(0.70, text=f"Exhaustive search for k={best_k}...")
        exh_sub, exh_rss = exhaustive_search(dft_matrix, exp_wn, exp_norm, best_k, config["poly_order"])
        sw_rss = sw_history[best_k - 1][1]
        if exh_rss < sw_rss - 1e-6:
            best_sel = exh_sub
        else:
            best_sel = best_sel_sw
    else:
        best_sel = best_sel_sw

    progress.progress(0.80, text="Final fit + bootstrap...")
    dc_best, bc_best, rss_best, resid_best = fit_lsq(dft_matrix[best_sel], exp_wn, exp_norm, config["poly_order"])
    recon_best = exp_norm - resid_best
    r2_best = 1 - rss_best / ss_tot if ss_tot > 0 else 0.0
    wt_best = dc_best / dc_best.sum() if dc_best.sum() > 0 else dc_best
    boot_w = run_bootstrap(dft_matrix, best_sel, exp_wn, exp_norm, int(config["n_bootstrap"]), block_len, config["poly_order"])
    sel_freq = np.mean(boot_w > 1e-4, axis=0)

    progress.progress(0.90, text="Peak residuals & sensitivity...")
    peaks, props = find_peaks(exp_norm, height=config["peak_height"], distance=config["peak_distance"], prominence=config["peak_prominence"])
    pk_wn = exp_wn[peaks] if len(peaks) > 0 else np.array([])
    pk_exp = exp_norm[peaks] if len(peaks) > 0 else np.array([])
    pk_fit = recon_best[peaks] if len(peaks) > 0 else np.array([])
    pk_res = pk_exp - pk_fit if len(peaks) > 0 else np.array([])

    # Scaling sensitivity
    sens_results = []
    for sf in config["scaling_sensitivity"]:
        shift = sf / config["scaling_factor"]
        dft_shifted = np.zeros_like(dft_matrix)
        for i in range(n_mol):
            dft_shifted[i] = np.interp(exp_wn, exp_wn * shift, dft_matrix[i], left=0, right=0)
        hist_s = forward_stepwise(dft_shifted, exp_wn, exp_norm, max_k=8, poly_order=config["poly_order"])
        if not hist_s:
            continue
        cv_s = [blocked_cv(dft_shifted, h[0], exp_wn, exp_norm, config["n_cv_blocks"], config["poly_order"]) for h in hist_s]
        bk = int(np.argmin(cv_s) + 1)
        sel_s = hist_s[bk - 1][0]
        dc_s, _, rss_s, _ = fit_lsq(dft_shifted[sel_s], exp_wn, exp_norm, config["poly_order"])
        r2_s = 1 - rss_s / ss_tot if ss_tot > 0 else 0.0
        top_i = sel_s[int(np.argmax(dc_s))]
        top_w = dc_s.max() / dc_s.sum() if dc_s.sum() > 0 else 0.0
        sens_results.append({"scale": sf, "k": bk, "r2": r2_s, "cv": cv_s[bk - 1],
                             "top_cid": molecules[top_i]["cid"], "top_w": top_w, "sel": sel_s})

    progress.progress(1.0, text="Done")

    # ---- RESULTS UI ----
    st.divider()
    st.header("Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Experimental points", n_wn)
    col2.metric("DFT candidates", n_mol)
    col3.metric("Effective n", n_eff)

    col1, col2, col3 = st.columns(3)
    col1.metric("Best k (CV)", best_k)
    col2.metric("R² (best model)", f"{r2_best:.4f}")
    col3.metric("R² (full model)", f"{r2_full:.4f}")

    with st.expander("Diagnostics"):
        st.markdown(f"- **Condition number:** {cond_num:.2e}")
        st.markdown(f"- **Max DFT-DFT cosine:** {max_cos:.4f} (CID {molecules[i_mc]['cid']} vs CID {molecules[j_mc]['cid']})")
        st.markdown(f"- **Pairs with cosine > 0.95:** {n_95}")
        st.markdown(f"- **Pairs with cosine > 0.90:** {n_90}")
        st.markdown(f"- **Hierarchical clusters (cut=0.10):** {len(set(cluster_labels))}")
        st.plotly_chart(make_gram_matrix_plot(molecules, dft_matrix), use_container_width=True)

    with st.expander("Ranking", expanded=True):
        st.plotly_chart(make_ranking_plot(molecules, pearson_sc, "Pearson on Smoothed First Derivative", "Pearson"), use_container_width=True)
        st.plotly_chart(make_ranking_plot(molecules, cosine_sc, "Cosine Similarity (Raw Spectra)", "Cosine"), use_container_width=True)

    with st.expander("Best Model", expanded=True):
        rows = []
        for i, idx in enumerate(best_sel):
            lo = float(np.percentile(boot_w[:, i], 2.5))
            hi = float(np.percentile(boot_w[:, i], 97.5))
            rows.append({
                "#": i + 1,
                "CID": molecules[idx]["cid"],
                "Name": molecules[idx]["name"],
                "Weight": f"{wt_best[i]:.1%}",
                "95% CI": f"[{lo:.1%}, {hi:.1%}]",
                "Sel%": f"{sel_freq[i]:.0%}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.plotly_chart(make_fit_plot(exp_wn, exp_norm, recon_best, best_sel, dft_matrix, molecules, best_k, r2_best), use_container_width=True)
        st.plotly_chart(make_bootstrap_plot(best_sel, boot_w, molecules), use_container_width=True)

    with st.expander("Model Selection"):
        ks = list(range(1, len(sw_history) + 1))
        r2s = [1 - h[1] / ss_tot for h in sw_history]
        st.plotly_chart(make_stepwise_plot(ks, r2s, bic_vals, cv_vals, best_k_bic, best_k_cv, n_eff, config["n_cv_blocks"]), use_container_width=True)
        stepwise_df = pd.DataFrame({
            "k": ks,
            "Added CID": [molecules[h[0][-1]]["cid"] for h in sw_history],
            "Added Name": [molecules[h[0][-1]]["name"] for h in sw_history],
            "RSS": [h[1] for h in sw_history],
            "R²": r2s,
            "BIC": bic_vals,
            "CV": cv_vals,
        })
        st.dataframe(stepwise_df, use_container_width=True)

    with st.expander("Peak-Resolved Residuals"):
        if len(pk_wn) > 0:
            st.markdown(f"Detected **{len(pk_wn)}** peaks")
            pk_df = pd.DataFrame({
                "Wavenumber": pk_wn,
                "Exp": pk_exp,
                "Fit": pk_fit,
                "Residual": pk_res,
                "Rel Err (%)": [abs(r / e) * 100 if abs(e) > 0.01 else 0 for r, e in zip(pk_res, pk_exp)],
            })
            st.dataframe(pk_df, use_container_width=True)
        else:
            st.markdown("No peaks detected with current criteria.")

    with st.expander("Scaling Factor Sensitivity"):
        if sens_results:
            st.plotly_chart(make_sensitivity_plot(sens_results), use_container_width=True)
            sens_df = pd.DataFrame({
                "Scale": [r["scale"] for r in sens_results],
                "Best k": [r["k"] for r in sens_results],
                "R²": [r["r2"] for r in sens_results],
                "Top CID": [r["top_cid"] for r in sens_results],
                "Top Wt": [r["top_w"] for r in sens_results],
            })
            st.dataframe(sens_df, use_container_width=True)
        else:
            st.markdown("No sensitivity results produced.")

    # Summary CSV
    summary_rows = []
    for i in range(n_mol):
        summary_rows.append({
            "CID": molecules[i]["cid"],
            "Name": molecules[i]["name"],
            "Pearson_Derivative": round(float(pearson_sc[i]), 5),
            "Cosine_Similarity": round(float(cosine_sc[i]), 5),
            "NNLS_Spectral_Weight_Pct": round(float(wt_full[i]) * 100, 3),
            "Pearson_Rank": int(np.where(pearson_rank == i)[0][0]) + 1,
            "Cosine_Rank": int(np.where(cosine_rank == i)[0][0]) + 1,
            "NNLS_Rank": int(np.where(nnls_rank == i)[0][0]) + 1,
            "In_Best_Model": "Yes" if i in best_sel else "No",
            "Cluster_ID": int(cluster_labels[i]),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("NNLS_Spectral_Weight_Pct", ascending=False)
    csv_buffer = io.StringIO()
    summary_df.to_csv(csv_buffer, index=False)

    st.divider()
    st.subheader("Download Results")
    col_dl1, col_dl2 = st.columns(2)
    col_dl1.download_button("Download summary CSV", csv_buffer.getvalue(), "spectral_analysis_summary.csv", "text/csv")

    try:
        pdf_buffer = build_pdf_report(
            exp_wn, exp_norm, molecules, dft_matrix, best_sel, best_k, r2_best, recon_best,
            pearson_sc, cosine_sc, sw_history, bic_vals, cv_vals, best_k_bic, best_k_cv,
            boot_w, sens_results, n_eff, config["n_cv_blocks"], config
        )
        col_dl2.download_button("Download PDF report", pdf_buffer, "spectral_analysis_report.pdf", "application/pdf")
    except Exception as e:
        st.warning(f"PDF generation failed: {e}")


if __name__ == "__main__":
    main()
