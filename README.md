# fluvialsubstrate

Riverbed substrate classification from UAV RGB orthophotos: annotate a few polygons in QGIS, train a
Random Forest on colour and texture features, map the whole reach, and assess the map with
design-based validation.

Authors: Tulio Soto Parra, David Farò, Guido Zolezzi

Paper: [*A Scalable Open Source Workflow for Riverbed Substrate Classification Using UAV Imagery*](https://www.mdpi.com/2072-4292/18/15/2529),
Remote Sensing 2026, 18, 2529 (open access).

## What it does

1. **Annotate**: draw polygons of homogeneous substrate (e.g. `dry-sand`, `wet-cobble`) on the orthophoto in QGIS.
2. **Train**: the orthophoto is divided into square blocks (default 50 px); colour and texture features of
   the blocks inside your polygons train a Random Forest, evaluated with spatial cross-validation.
3. **Map**: every block of the orthophoto is classified; a confidence map shows where the model is unsure.
4. **Validate**: a random sample of map blocks is labelled blind in QGIS, giving unbiased accuracies and class
   areas with confidence intervals.
5. **Improve**: a diagnostics report says which classes need more or better polygons and where to annotate next.

Everything runs from one notebook, [substrate.ipynb](substrate.ipynb), on a normal laptop (no GPU).
See [CHANGELOG.md](CHANGELOG.md) for the changes since the paper.

## Installation

You need [Git](https://git-scm.com/downloads), Python 3.10–3.12 and [QGIS](https://qgis.org) (for drawing
polygons and labelling validation samples). Choose **one** of the two options below.

### Option A: conda (recommended, same on Windows, macOS and Linux)

Install [Miniforge](https://github.com/conda-forge/miniforge) (or Anaconda/Miniconda), then open a terminal
(Windows: *Miniforge Prompt*):

```bash
git clone https://github.com/tsotop/fluvialsubstrate.git
cd fluvialsubstrate
conda env create -f environment.yml
conda activate fluvialsubstrate
```

### Option B: pip with a virtual environment

macOS / Linux (Terminal):

```bash
git clone https://github.com/tsotop/fluvialsubstrate.git
cd fluvialsubstrate
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows (PowerShell; Python from [python.org](https://www.python.org/downloads/windows/)):

```powershell
git clone https://github.com/tsotop/fluvialsubstrate.git
cd fluvialsubstrate
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell refuses to run the activation script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use `.venv\Scripts\activate.bat` in the
Command Prompt.

### Check the installation

```bash
python -m unittest discover tests
```

All tests should pass (`OK`).

## Quick start

```bash
jupyter lab substrate.ipynb
```

(or open the notebook in VS Code and select the `fluvialsubstrate` / `.venv` interpreter). Then:

1. Copy your orthophoto to `data/orthophoto.tif` (or set its path in Step 2).
2. Run the notebook from top to bottom; it pauses naturally at the two QGIS steps (drawing polygons
   after Step 3, labelling validation samples after Step 7).

The full step-by-step guide, including the QGIS parts, is in [USAGE.md](USAGE.md).

## Project structure

```text
fluvialsubstrate/
|-- substrate.ipynb       # The workflow (Steps 1-9)
|-- README.md, USAGE.md, CHANGELOG.md, LICENSE
|-- requirements.txt      # pip dependencies
|-- environment.yml       # conda environment
|-- configs/default.yaml  # Default settings (also set in notebook Step 2)
|-- src/                  # Workflow code
|   |-- annotation.py     # Seed points and annotation GeoPackage
|   |-- block_features.py # Block grid and colour/texture features
|   |-- features.py       # Training data from the annotation polygons
|   |-- modeling.py       # Feature selection, cross-validation, final model
|   |-- inference.py      # Full-map classification
|   |-- postprocess.py    # Class map, confidence, edge filling
|   |-- validation.py     # Validation sample and accuracy/area report
|   |-- diagnostics.py    # Tips to improve the model
|   `-- ml_utils.py, roi_utils.py, plotting.py
|-- tests/                # Automated tests
|-- benchmarks/           # Benchmark on the study rivers (for development)
|-- data/                 # Your input data (not versioned)
`-- outputs/              # Results (not versioned)
```

## Citation

If you use this workflow, please cite:

> Soto Parra, T.; Farò, D.; Zolezzi, G. A Scalable Open Source Workflow for Riverbed Substrate
> Classification Using UAV Imagery. *Remote Sens.* **2026**, *18*, 2529. https://doi.org/10.3390/rs18152529

## License

MIT License, see [LICENSE](LICENSE).
