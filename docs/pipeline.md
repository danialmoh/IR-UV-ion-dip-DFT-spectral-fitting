# Analysis Pipeline Overview

This document describes the full spectral analysis pipeline implemented in `spectral_analysis_v3.py` and exposed through the Streamlit app in `spectral_analysis_app.py`.

---

## 1. Data loading and normalization

- Load the experimental IR-UV ion-dip spectrum from CSV
- Trim to the selected wavenumber range (default: 400–1600 cm⁻¹)
- Normalize the intensity to the [0, 1] range

## 2. DFT loading and interpolation

- Load each DFT convoluted spectrum (`CID_*_scaled_*.csv`)
- Interpolate each DFT spectrum onto the experimental wavenumber grid using linear interpolation
- Stack all DFT spectra into a design matrix

## 3. Diagnostics

- **Collinearity matrix**: Pearson correlation between all DFT spectra, shown as a heatmap
- **Clustering**: hierarchical clustering of DFT spectra to reveal groups of similar isomers

## 4. Similarity ranking

Each DFT spectrum is scored against the experimental spectrum using two metrics:

- **Pearson derivative correlation**: correlation of the first-derivative spectra, emphasizing peak positions
- **Cosine similarity**: angle between the experimental and DFT intensity vectors

These rankings help identify which single isomers match the experiment best.

## 5. Full NNLS fit with baseline

A non-negative least squares (NNLS) fit is performed with a polynomial baseline added to the design matrix:

```
exp ≈ Σ w_i * DFT_i + baseline
w_i ≥ 0
```

The baseline order is configurable (default: 1, i.e. linear).

## 6. Forward stepwise subset selection

Starting from an empty model, the algorithm repeatedly adds the DFT spectrum that most improves the fit (largest reduction in residual sum of squares) until the maximum allowed subset size is reached.

## 7. Model selection

For each candidate subset size, two criteria are computed:

- **Bayesian Information Criterion (BIC)** with an effective number of independent points based on the average autocorrelation of the residuals
- **Blocked k-fold cross-validation (CV)** error, where the data are divided into contiguous blocks to preserve spectral correlation structure

The model size with the best BIC and the best CV score are reported, and a final model is selected.

## 8. Exhaustive search for small subsets

For subset sizes up to a small limit (default: 5), all combinations are evaluated by NNLS. This guarantees that the optimal small combination is found, at the cost of exponential growth.

## 9. Final fit and bootstrap confidence intervals

Once the final subset is selected, the fit is recomputed and block-bootstrap resampling is used to estimate confidence intervals for each spectral weight. The blocks preserve local correlations in the residuals.

## 10. Peak-resolved residual analysis

The residual spectrum is analyzed to identify regions where the model systematically under- or over-predicts the experimental intensity. Peak detection helps flag important missing modes or mismatched bands.

## 11. Sensitivity analysis

Two sensitivity checks are performed:

- **DFT scaling factor**: the final fit is re-evaluated at a grid of scale factors around the nominal value to show how robust the selected isomers are to scaling
- **Experimental smoothing**: the experimental spectrum is smoothed with a Savitzky–Golay filter and the fit is re-run to assess stability against noise

## 12. Outputs

- `summary.csv` — weights, selection criteria, and diagnostics for the final model and alternatives
- `*_report.pdf` — a multi-page PDF containing:
  - Rankings
  - Collinearity heatmap
  - Full NNLS fit
  - Stepwise selection path
  - BIC and CV curves
  - Final fit with bootstrap confidence intervals
  - Residual analysis
  - Sensitivity plots

---

## References / theory

- NNLS: Lawson & Hanson, *Solving Least Squares Problems* (1974, 1995)
- BIC: Schwarz, G. E. *The Annals of Statistics* (1978)
- Blocked CV / bootstrap: preserves correlation structure in ordered spectral data
