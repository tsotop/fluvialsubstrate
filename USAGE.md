# fluvialsubstrate - Usage Instructions

This guide explains how to execute the notebook workflow in [substrate.ipynb](substrate.ipynb).

## 1. Prerequisites

- Python 3.10+
- QGIS (recommended for annotation and validation labeling)

## 2. Environment setup

```bash
cd /Users/tsotop/Unitn/Substrate/final/Jupyter
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Prepare inputs

Provide:

- Orthophoto GeoTIFF (default: [data/orthophoto.tif](data/orthophoto.tif))
- Annotation GeoPackage path (default: `data/sampling_data.gpkg`)

All paths are defined in Step 2 of [substrate.ipynb](substrate.ipynb).

## 4. Run the notebook sequentially

Execute all steps from top to bottom.

1. Install dependencies
2. Configure paths and parameters
3. Generate annotation sample
4. Extract features
5. Train model
6. Classify map
7. Generate validation samples
8. Compute final accuracy

## 5. Step 3 details: annotations and ROI

Step 3 creates a GeoPackage with:

- points
- annotations
- roi

How to use:

- Fill `annotations` with substrate polygons.
- Optionally draw ROI polygons in `roi`.

Behavior:

- ROI not empty: downstream processing runs only inside ROI.
- ROI empty: full-domain processing.

## 6. Outputs by stage

Training and inference outputs:

- [outputs/features](outputs/features): extracted block features
- [outputs/models](outputs/models): model artifacts and class mapping
- [outputs/maps](outputs/maps): classified raster products

Validation outputs (Step 7):

- [outputs/validation/validation_true_annotations.gpkg](outputs/validation/validation_true_annotations.gpkg)
- [outputs/validation/validation_modeled.gpkg](outputs/validation/validation_modeled.gpkg)

## 7. Validation workflow in QGIS

1. Open `validation_true_annotations.gpkg`.
2. Populate the `true_label` field for each polygon.
3. Keep `validation_modeled.gpkg` unchanged.
4. Run Step 8 in the notebook.

## 8. Troubleshooting

- No extracted samples: verify `annotations` contains valid polygons and labels.
- Empty map output inside ROI: verify ROI overlaps the orthophoto domain.
- No validation samples: verify Step 6 produced a valid classified map.

## 9. Good practice

- Commit code/notebook/docs frequently.
- Keep data and outputs local; do not version large generated artifacts.
