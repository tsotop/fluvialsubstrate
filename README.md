# fluvialsubstrate

Fluvial substrate classification framework based on orthophotos, with a notebook-first workflow for annotation, training, inference, and validation.

Authors: Tulio Soto Parra, David Faro, Guido Zolezzi

From the paper: A Scalable Open-Source Workflow for Riverbed Substrate Classification Using UAV Imagery
Now as preprint! https://doi.org/10.20944/preprints202605.1410.v1


## Description

fluvialsubstrate is an end-to-end computational workflow to estimate river substrate classes from orthophotos.

The framework combines:

- GIS-assisted polygon annotation
- Block-based spectral and texture feature extraction
- Supervised machine learning classification
- Stratified validation sampling and accuracy assessment

It is designed to be practical for research and operational mapping, using interoperable geospatial formats (GeoPackage, GeoTIFF) and a single main notebook pipeline.

## Features

- Notebook-first workflow: one main notebook for the full pipeline.
- Model flexibility: RandomForest, XGBoost, and LightGBM support.
- ROI-aware processing: optional ROI layer constrains extraction, inference, and validation.
- GIS-ready outputs: annotation, map, and validation products in standard formats.
- Reproducible setup: requirements file and clear folder structure.

## Project structure

```text
fluvialsubstrate/
|
|-- substrate.ipynb              # Main notebook pipeline (Steps 1-8)
|-- README.md                    # Project overview
|-- USAGE.md                     # Detailed usage guide
|-- requirements.txt             # Python dependencies
|-- .gitignore
|
|-- src/                         # Core pipeline modules
|   |-- annotation.py
|   |-- features.py
|   |-- feature_extractor_masterv2.py
|   |-- modeling.py
|   |-- inference.py
|   |-- validation.py
|   |-- roi_utils.py
|   |-- ml_utils.py
|   `-- plotting.py
|
|-- data/                        # Input data (local)
|-- configs/                     # Optional config files
`-- outputs/                     # Generated artifacts (local)
```

## Installation

This project is developed for Python 3.10+.

```bash
git clone https://github.com/<your-username>/fluvialsubstrate.git
cd fluvialsubstrate
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the workflow through [substrate.ipynb](substrate.ipynb):

1. Install packages
2. Configure paths and parameters
3. Generate annotation sample
4. Extract features
5. Train model
6. Classify map
7. Generate validation set
8. Compute final accuracy

For detailed step-by-step instructions, see [USAGE.md](USAGE.md).

## Configuration

Configuration is notebook-based (Step 2 in [substrate.ipynb](substrate.ipynb)).

Main sections:

- paths: orthophoto, labels, outputs
- features: block size, tile size, feature families
- model: model type, class weighting, feature selection settings
- validation: sample size and confidence settings

## Optional ROI workflow

Step 3 creates a GeoPackage with:

- points
- annotations
- roi

ROI behavior:

- If roi contains polygons, downstream processing is restricted to ROI.
- If roi is empty, the full domain is processed.

ROI is applied in:

- feature extraction
- map inference
- validation sample generation

## Validation outputs

Step 7 writes two files under outputs/validation:

- validation_true_annotations.gpkg
- validation_modeled.gpkg

Step 8 reads both files to compute design-based and standard classification metrics.

## GitHub setup

Create an empty GitHub repository named fluvialsubstrate, then run:

```bash
cd fluvialsubstrate
git remote add origin https://github.com/<your-username>/fluvialsubstrate.git
git push -u origin main
```

## Notes

- Large data and outputs are intentionally ignored by .gitignore.
- Keep data and outputs local; version source code, notebook, and docs.
