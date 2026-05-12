# fluvialsubstrate

A robust, user-friendly workflow for fluvial substrate class estimation from orthophotos.

Authors: Tulio Soto Parra, David Faro, Guido Zolezzi

Reference: A robust, user-friendly tool for accurate fluvial substrate class estimation.

## Overview

This project provides an end-to-end pipeline to:

- Build annotation samples for training.
- Extract spectral and texture features from orthophotos.
- Train a classifier (RandomForest, XGBoost, or LightGBM).
- Generate classified substrate maps.
- Create validation samples and compute final accuracy metrics.

The workflow is designed for reproducibility and GIS interoperability through GeoPackage and GeoTIFF outputs.

## Optional ROI workflow

Step 3 generates a GeoPackage with three layers:

- points
- annotations
- roi

The roi layer is optional.

- If roi contains one or more polygons, downstream steps run only inside ROI.
- If roi is empty, the full orthophoto domain is used.

ROI filtering is applied in:

- Feature extraction
- Map inference
- Validation sample generation

## Project structure

- substrate.ipynb: Main notebook pipeline (Steps 1-8).
- src/: Core Python modules.
- data/: Inputs (orthophoto, annotation GeoPackage).
- outputs/: Models, maps, and validation artifacts.
- requirements.txt: Python dependencies.

## Quick start

1. Create and activate a Python environment.
2. Install dependencies.
3. Open substrate.ipynb.
4. Run steps sequentially.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Notebook pipeline

1. Install Required Packages
2. Configuration
3. Generate Annotation Sample
4. Extract Features
5. Fit Model
6. Classify Map
7. Generate Validation Set
8. Final Validation

## Validation outputs

Step 7 writes two files in outputs/validation:

- validation_true_annotations.gpkg: for manual true labels.
- validation_modeled.gpkg: modeled labels and stratum metadata.

Step 8 reads both files to compute accuracy metrics.

## GitHub setup for fluvialsubstrate

From this folder, initialize and push the repository:

```bash
cd /Users/tsotop/Unitn/Substrate/final/Jupyter
git init
git branch -M main
git add .
git commit -m "Initial commit: fluvialsubstrate pipeline"
```

Create a new empty GitHub repository named fluvialsubstrate, then connect and push:

```bash
git remote add origin https://github.com/<your-username>/fluvialsubstrate.git
git push -u origin main
```

If you use GitHub CLI:

```bash
gh repo create fluvialsubstrate --public --source=. --remote=origin --push
```

## Notes

- Large rasters and generated outputs should not be committed.
- Use the provided .gitignore to keep the repository clean.
