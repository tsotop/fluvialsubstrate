# fluvialsubstrate Usage Instructions

This document explains how to run the full workflow in [substrate.ipynb](substrate.ipynb).

## 1. Prerequisites

- Python 3.10+ recommended
- QGIS (for annotation and validation labeling)

## 2. Environment setup

From the project root:

```bash
cd /Users/tsotop/Unitn/Substrate/final/Jupyter
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Input data

Expected files:

- Orthophoto raster at [data/orthophoto.tif](data/orthophoto.tif)
- Annotation GeoPackage path configured as `data/sampling_data.gpkg`

You can edit paths in Step 2 of [substrate.ipynb](substrate.ipynb).

## 4. Run the notebook

Open [substrate.ipynb](substrate.ipynb) and run steps in order.

### Step 3: Annotation sample and optional ROI

Step 3 creates `sampling_data.gpkg` with these layers:

- `points`: reference points for manual annotation
- `annotations`: polygon labels used for training (`substrate` field)
- `roi`: optional ROI polygons

ROI behavior:

- If `roi` contains polygons, downstream processing is restricted to ROI.
- If `roi` is empty, processing runs on the full domain.

## 5. Training and inference outputs

After Steps 4 to 6, key outputs are created under [outputs](outputs):

- [outputs/features](outputs/features): extracted features
- [outputs/models](outputs/models): trained model + feature list + class mapping
- [outputs/maps/classified_map.tif](outputs/maps/classified_map.tif): classified map

## 6. Validation workflow

Step 7 generates two validation files in [outputs/validation](outputs/validation):

- [outputs/validation/validation_true_annotations.gpkg](outputs/validation/validation_true_annotations.gpkg)
- [outputs/validation/validation_modeled.gpkg](outputs/validation/validation_modeled.gpkg)

How to use them:

1. Open `validation_true_annotations.gpkg` in QGIS.
2. Fill `true_label` for each sample polygon.
3. Keep `validation_modeled.gpkg` unchanged.
4. Run Step 8 to compute metrics.

## 7. Common checks

- No training polygons: verify `annotations` has valid substrate labels.
- Empty ROI effect: if ROI is empty, full-domain behavior is expected.
- No validation samples: verify classified map exists and ROI overlaps mapped area.

## 8. Reproducibility tips

- Keep `configs/default.yaml` and notebook config aligned.
- Commit code/config changes before rerunning full pipelines.
- Avoid committing local data and generated outputs; `.gitignore` already excludes them.
