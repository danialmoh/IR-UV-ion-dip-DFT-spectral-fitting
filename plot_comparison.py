"""
Offset comparison plot: Experimental m/z 140 IRMPD vs scaled DFT spectra (b3lyp/def2-tzvp).
Each page has 6 panels. Each panel shows:
  - Experimental spectrum (gray fill)
  - DFT convoluted spectrum (colored line)
  - DFT stick spectrum (colored stems)
  - CID structure image on the right
Output: multi-page PDF
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
from matplotlib.image import imread
from matplotlib.gridspec import GridSpec

# === PATHS ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_FILE = os.path.join(BASE_DIR, "mass_channel_IR_spectra_wn401-1576_2ch-negativenotcropped.csv")
DFT_DIR = os.path.join(BASE_DIR, "b3lyp_def2-tzvp", "scaled")
IMG_DIR = os.path.join(BASE_DIR, "PubChem_search_images-2")
OUTPUT_PDF = os.path.join(BASE_DIR, "b3lyp_def2-tzvp", "plots", "140_comparison_offset.pdf")

# === SETTINGS ===
PANELS_PER_PAGE = 6
WAVENUMBER_RANGE = (400, 1600)  # cm-1
EXP_COLOR = "0.4"  # gray
EXP_ALPHA = 0.35
OFFSET_FACTOR = 0.35  # vertical offset between exp and DFT within a panel

# Color palette for DFT spectra (one color per molecule, cycling a colormap)
COLORMAP = plt.colormaps["tab10"]

# === Load experimental spectrum ===
exp_df = pd.read_csv(EXP_FILE)
exp_wn = exp_df["Wavenumber"].values
exp_intensity = exp_df["m/z 140.0 -ln(depl)"].values

# Normalize experimental to [0, 1]
exp_max = exp_intensity.max()
if exp_max > 0:
    exp_norm = exp_intensity / exp_max
else:
    exp_norm = exp_intensity

# Pre-trim experimental to display range
exp_mask = (exp_wn >= WAVENUMBER_RANGE[0]) & (exp_wn <= WAVENUMBER_RANGE[1])
exp_wn = exp_wn[exp_mask]
exp_norm = exp_norm[exp_mask]
exp_min = exp_norm.min()

# === Collect DFT file pairs ===
convoluted_files = sorted(glob.glob(os.path.join(DFT_DIR, "CID_*_scaled_0.95.csv")))

molecules = []
for conv_file in convoluted_files:
    basename = os.path.basename(conv_file)
    # Extract CID number
    cid_match = re.match(r"CID_(\d+)_(.+)_scaled_0\.95\.csv", basename)
    if not cid_match:
        continue
    cid = cid_match.group(1)
    name = cid_match.group(2).replace("_", " ").replace("-", "-")
    stick_file = conv_file.replace("_scaled_0.95.csv", "_scaled_0.95_sticks.csv")
    img_file = os.path.join(IMG_DIR, f"CID_{cid}.png")

    if not os.path.exists(stick_file):
        continue

    molecules.append({
        "cid": cid,
        "name": name,
        "conv_file": conv_file,
        "stick_file": stick_file,
        "img_file": img_file if os.path.exists(img_file) else None,
    })

print(f"Found {len(molecules)} molecules to plot.")

# === Create output directory ===
os.makedirs(os.path.dirname(OUTPUT_PDF), exist_ok=True)

# === Export combined CSV (exp + all DFT convoluted on shared wavenumber grid) ===
OUTPUT_CSV = os.path.join(BASE_DIR, "b3lyp_def2-tzvp", "plots", "140_exp_vs_dft_scaled_0.95_all.csv")

# Build DataFrame with experimental as first column
combined_df = pd.DataFrame({"Wavenumber_cm-1": exp_wn, "Exp_mz140_-ln(depl)": exp_norm})

for mol in molecules:
    dft_df = pd.read_csv(mol["conv_file"], comment="#", header=None,
                         names=["Wavenumber", "Fundamentals", "Overtones",
                                "Combinations", "Total"])
    dft_wn_raw = dft_df["Wavenumber"].values
    dft_total_raw = dft_df["Total"].values
    # Interpolate DFT onto experimental wavenumber grid
    dft_interp = np.interp(exp_wn, dft_wn_raw, dft_total_raw)
    col_name = f"CID_{mol['cid']}_{mol['name']}_scaled_0.95"
    combined_df[col_name] = dft_interp

combined_df.to_csv(OUTPUT_CSV, index=False)
print(f"Combined CSV saved to:\n{OUTPUT_CSV}")

# === Plot ===
n_pages = int(np.ceil(len(molecules) / PANELS_PER_PAGE))

with PdfPages(OUTPUT_PDF) as pdf:
    for page_idx in range(n_pages):
        start = page_idx * PANELS_PER_PAGE
        end = min(start + PANELS_PER_PAGE, len(molecules))
        page_mols = molecules[start:end]
        n_panels = len(page_mols)

        fig = plt.figure(figsize=(11, 2.2 * n_panels + 0.8))
        # GridSpec: n_panels rows, 2 columns (spectrum | structure)
        gs = GridSpec(n_panels, 2, figure=fig, width_ratios=[4, 1],
                      hspace=0.35, wspace=0.05,
                      left=0.07, right=0.97, top=0.95, bottom=0.06)

        for i, mol in enumerate(page_mols):
            color = COLORMAP(i % 10)

            # --- Load DFT convoluted ---
            dft_df = pd.read_csv(mol["conv_file"], comment="#", header=None,
                                 names=["Wavenumber", "Fundamentals", "Overtones",
                                        "Combinations", "Total"])
            dft_wn = dft_df["Wavenumber"].values
            dft_total = dft_df["Total"].values
            # Trim to display range and normalize
            in_range_dft = (dft_wn >= WAVENUMBER_RANGE[0]) & (dft_wn <= WAVENUMBER_RANGE[1])
            dft_wn = dft_wn[in_range_dft]
            dft_total = dft_total[in_range_dft]
            dft_max = dft_total.max() if dft_total.max() > 0 else 1.0
            dft_norm = dft_total / dft_max

            # --- Load DFT sticks ---
            stick_df = pd.read_csv(mol["stick_file"], comment="#", header=None,
                                   names=["Mode", "Frequency", "Intensity", "Type"])
            # Keep fundamentals + overtones/combinations above 10% of global max
            stick_freq_all = stick_df["Frequency"].values
            stick_int_all = stick_df["Intensity"].values
            stick_type = stick_df["Type"].values
            stick_max_global = stick_int_all.max() if stick_int_all.max() > 0 else 1.0
            threshold = 0.10 * stick_max_global
            keep = (stick_type == "fundamental") | (stick_int_all >= threshold)
            stick_freq = stick_freq_all[keep]
            stick_int = stick_int_all[keep]
            # Normalize sticks using global max
            stick_norm = stick_int / stick_max_global

            # --- Spectrum axis ---
            ax = fig.add_subplot(gs[i, 0])

            # Plot experimental (bottom, gray filled from axis bottom)
            ax.fill_between(exp_wn, exp_min - 0.05, exp_norm, color=EXP_COLOR,
                            alpha=EXP_ALPHA, label="Exp m/z 140", zorder=1)
            ax.plot(exp_wn, exp_norm, color="0.2", lw=0.8, zorder=2)

            # Plot DFT convoluted (offset up)
            offset = OFFSET_FACTOR
            ax.plot(dft_wn, dft_norm + offset, color=color, lw=1.2,
                    label="DFT (conv.)", zorder=4)
            ax.fill_between(dft_wn, offset, dft_norm + offset,
                            color=color, alpha=0.15, zorder=3)

            # Plot DFT sticks on the same baseline as experimental (from axis bottom)
            # Filter sticks within range
            mask = (stick_freq >= WAVENUMBER_RANGE[0]) & (stick_freq <= WAVENUMBER_RANGE[1])
            ax.vlines(stick_freq[mask], exp_min - 0.05, stick_norm[mask],
                      colors=color, alpha=0.55, lw=0.7, zorder=5)

            # Formatting
            ax.set_xlim(WAVENUMBER_RANGE)
            ax.set_ylim(exp_min - 0.05, 1.0 + offset + 0.1)
            ax.set_yticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)

            # Molecule name label
            label_text = mol["name"]
            if len(label_text) > 40:
                label_text = label_text[:37] + "..."
            ax.text(0.01, 0.92, f"CID {mol['cid']}", transform=ax.transAxes,
                    fontsize=7, fontweight="bold", va="top", color=color)
            ax.text(0.01, 0.78, label_text, transform=ax.transAxes,
                    fontsize=6, va="top", color=color, style="italic")

            if i == n_panels - 1:
                ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=9)
            else:
                ax.set_xticklabels([])

            # --- Structure image axis ---
            ax_img = fig.add_subplot(gs[i, 1])
            ax_img.axis("off")
            if mol["img_file"]:
                try:
                    img = imread(mol["img_file"])
                    ax_img.imshow(img, aspect="equal")
                except Exception:
                    ax_img.text(0.5, 0.5, "No image", ha="center", va="center",
                                transform=ax_img.transAxes, fontsize=8)
            else:
                ax_img.text(0.5, 0.5, "No image", ha="center", va="center",
                            transform=ax_img.transAxes, fontsize=8)

        # Page title
        fig.suptitle(f"m/z 140 Experimental vs B3LYP/def2-TZVP (page {page_idx+1}/{n_pages})",
                     fontsize=11, fontweight="bold", y=0.99)

        pdf.savefig(fig, dpi=100)
        plt.close(fig)

print(f"Done! PDF saved to:\n{OUTPUT_PDF}")
