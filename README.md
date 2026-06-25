# Spectral Analysis for C11H8 IR-UV Ion-Dip Spectroscopy

This repository contains the Python analysis pipeline and Streamlit web app used to compare experimental IR-UV ion-dip spectra of C11H8 isomers with DFT-computed vibrational spectra (B3LYP/def2-TZVP and B3LYP/N07D).

The repository holds **code only** — no experimental data, DFT output files, or generated plots are tracked by Git.

---

## What is included

| File / folder | Purpose |
| --- | --- |
| `spectral_analysis.py` | Original quantitative comparison pipeline: cosine ranking, NNLS fitting, sequential addition, BIC model selection, residual analysis |
| `spectral_analysis_v2.py` | Extended pipeline with diagnostics, ranking, exhaustive search, and PDF reporting |
| `spectral_analysis_v3.py` | Full production pipeline with blocked cross-validation, bootstrap confidence intervals, peak-resolved residuals, and sensitivity analysis |
| `spectral_analysis_app.py` | **Streamlit app** that wraps the v3 pipeline with interactive uploads, parameter controls, and downloadable CSV/PDF reports |
| `plot_comparison.py` | Multi-page offset comparison plots of experimental vs. DFT spectra |
| `requirements.txt` | Python dependencies |
| `docs/` | Usage guides and explanation of the data formats / analysis pipeline |
| `.gitignore` | Ensures data files and generated outputs are never committed |

## What is NOT included

- Experimental IR-UV spectra (large CSV files)
- DFT `.out` files and computed spectra (`b3lyp_def2-tzvp/`, `b3lyp_n07d/`, `IR_results/`)
- Structure images and PDF reports
- Any generated plots or summary CSVs

These files are excluded via `.gitignore` so that the repository stays lightweight and focused on the analysis code.

---

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/IR-UV-ion-dip-DFT-spectral-fitting.git
cd IR-UV-ion-dip-DFT-spectral-fitting
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run spectral_analysis_app.py
```

Then upload your experimental CSV and the DFT scaled CSV files through the app UI, adjust parameters in the sidebar, and click **Run Analysis**.

---

## Running the command-line pipeline

To run the v3 pipeline from the terminal (after placing the required data files locally):

```bash
python spectral_analysis_v3.py
```

See `docs/usage.md` for detailed instructions on file formats and expected folder layout.

---

## Analysis pipeline overview

The v3 pipeline performs the following steps:

1. Load and normalize the experimental spectrum
2. Load and interpolate DFT spectra onto the experimental grid
3. Diagnostics: collinearity matrix and clustering of DFT spectra
4. Similarity ranking using Pearson derivative and cosine metrics
5. Full non-negative least squares (NNLS) fit with polynomial baseline
6. Forward stepwise subset selection
7. Model selection via BIC and blocked k-fold cross-validation
8. Exhaustive search for small subset sizes
9. Final fit with block-bootstrap confidence intervals
10. Peak-resolved residual analysis
11. Sensitivity analysis to DFT scaling factor and experimental smoothing
12. Export summary CSV and PDF report

More details are in `docs/pipeline.md`.

---

## DFT spectrum generation

The convoluted scaled/unscaled DFT spectra used by the pipeline are produced from ORCA `.out` files using separate generator scripts. Those scripts live in the project folders that contain the DFT calculations and are not duplicated here, but they follow the same logic:

- Parse the `IR SPECTRUM` and `OVERTONES AND COMBINATION BANDS` sections
- Convolute with a Gaussian whose FWHM is 0.54% of the wavenumber
- Apply a basis-set-specific scale factor (def2-TZVP: 0.95; N07D: 0.97)
- Write `CID_<cid>_<name>_scaled_<factor>.csv` plus `_sticks.csv` counterparts

---

## License

[Add your license here, e.g. MIT]

---

## Contact / citation

[Add your contact information or citation text here]
