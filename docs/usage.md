# Usage Guide

This guide explains how to run the scripts and the Streamlit app.

---

## Streamlit app (`spectral_analysis_app.py`)

The easiest way to use the analysis pipeline is through the interactive app.

### 1. Start the app

```bash
streamlit run spectral_analysis_app.py
```

The app opens in your browser at `http://localhost:8501`.

### 2. Upload data

- **Experimental spectrum**: a CSV with at least a wavenumber column and an intensity column (e.g. `m/z 140.0 -ln(depl)`).
- **DFT spectra**: multiple CSV files named `CID_<cid>_<name>_scaled_<factor>.csv`. Upload all isomers you want to compare.

### 3. Adjust parameters

Use the sidebar to set:

- Wavenumber range (default: 400–1600 cm⁻¹)
- FWHM for the analysis (default: 10 cm⁻¹)
- DFT scale factor (default: 0.95)
- Bootstrap iterations (default: 1000)
- Maximum forward stepwise subset size (default: 12)
- Maximum exhaustive search subset size (default: 5)
- Number of CV blocks (default: 10)
- Polynomial baseline order (default: 1)
- Peak detection thresholds

### 4. Run and download

Click **Run Analysis**. The app shows interactive plots and tables, and provides download buttons for:

- A summary CSV
- A PDF report

---

## Command-line scripts

### `spectral_analysis_v3.py`

Run the full v3 pipeline from the terminal:

```bash
python spectral_analysis_v3.py
```

Expected local layout (data is not tracked by Git, but the script reads from these relative paths):

```
.
mass_channel_IR_spectra_wn401-1576_2ch.csv   # experimental spectrum
b3lyp_def2-tzvp/
  scaled/
    CID_*.csv
```

Outputs are written to `b3lyp_def2-tzvp/plots/`.

### `plot_comparison.py`

Generate multi-page offset comparison plots:

```bash
python plot_comparison.py
```

It reads `b3lyp_def2-tzvp/scaled/CID_*_scaled_0.95.csv` and writes `b3lyp_def2-tzvp/plots/140_comparison_offset.pdf` plus a combined CSV.

---

## Tips

- The pipeline is computationally intensive for many isomers. Reduce the number of bootstrap iterations or the maximum exhaustive search size for faster exploration.
- Use the Streamlit app for interactive tuning; use the command-line scripts for batch, reproducible runs.
