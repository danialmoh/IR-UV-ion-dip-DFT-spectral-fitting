# Data Formats

This document describes the file formats the pipeline expects. No data files are stored in the repository; you must supply them locally.

---

## Experimental spectrum

A CSV file with at least these columns:

- `Wavenumber` — wavenumber in cm⁻¹
- One or more intensity columns, e.g. `m/z 140.0 -ln(depl)`

Example (first few rows):

```csv
Wavenumber,m/z 140.0 -ln(depl),m/z 152.0 -ln(depl)
401.0,0.123,0.045
402.0,0.145,0.067
...
```

The Streamlit app lets you select which intensity column to use.

---

## DFT convoluted spectra

Each isomer has a convoluted spectrum file and a stick file.

### Convoluted spectrum

File name pattern:

```
CID_<cid>_<molecule_name>_scaled_<factor>.csv
```

Example:

```
CID_10582869_1-ethynyl-2-propadienylbenzene_scaled_0.95.csv
```

The first line is a comment header:

```csv
# Wavenumber(cm-1),Fundamentals(km/mol),Overtones(km/mol),Combinations(km/mol),Total(km/mol)
```

Subsequent rows contain comma-separated values:

```csv
350.00,0.000000,0.000000,0.000000,0.000000
350.73,0.000011,0.000000,0.000000,0.000011
...
```

The `Total` column is the sum of fundamentals, overtones, and combinations and is the column normally used for fitting.

### Stick spectrum

Each convoluted file has a corresponding `_sticks.csv` file with the underlying stick transitions:

```
CID_10582869_1-ethynyl-2-propadienylbenzene_scaled_0.95_sticks.csv
```

Header:

```csv
# Mode,Frequency(cm-1),Intensity(km/mol),Type
```

Rows look like:

```csv
7,358.03,1.0129,fundamental
7+7,716.06,0.0158,overtone
7+8,755.11,0.1340,combination
...
```

---

## How the DFT files are generated

The convoluted CSVs are produced from ORCA frequency output files (`.out`) by a generator script. The generator:

1. Parses the `IR SPECTRUM` section for fundamental modes
2. Parses the `OVERTONES AND COMBINATION BANDS` section for overtones and combinations
3. Convolutes each stick transition with a Gaussian lineshape whose FWHM is 0.54% of the transition wavenumber
4. Scales the frequencies by the basis-set factor (def2-TZVP: 0.95; N07D: 0.97)
5. Writes the unscaled and scaled convoluted spectra plus stick files

These generator scripts are kept in the directories that contain the DFT calculations, not in this repository, because they are tightly coupled to the `.out` file organization there.

---

## Outputs

When you run the analysis pipeline or the Streamlit app, several outputs are produced:

- `summary.csv` — model weights, selection criteria, diagnostics
- `*_report.pdf` — multi-page PDF with rankings, fits, model selection, bootstrap results, residuals, and sensitivity analysis
- Various `.png` diagnostic plots (optional, depending on the script)

All generated outputs are excluded from Git by `.gitignore`.
