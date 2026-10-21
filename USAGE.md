# How to use fluvialsubstrate

The workflow is the notebook [substrate.ipynb](substrate.ipynb). Run it from top to bottom; two steps
happen in QGIS. Installation: see [README.md](README.md#installation).

## Before you start

- **Orthophoto**: an RGB GeoTIFF (8-bit) with a projected coordinate system (metres), e.g. from
  Agisoft Metashape or OpenDroneMap. Areas outside the mosaic may be transparent (alpha band), black or white.
- **Class list**: decide your substrate classes before annotating and write the names consistently,
  e.g. `dry-sand`, `wet-sand`, `dry-cobble`, `wet-cobble`, `dry-boulder`, `vegetation`. Annotate wet and dry
  variants of the same material as separate classes.
- **Block size**: the map has one class per square block. The default, 50 px, is about 1 m at 2 cm
  resolution; aim for roughly 0.75–1 m blocks.

## Step 1–2: Install and configure

Step 1 installs the requirements (skip it if you followed the installation in the README).
Step 2 holds all settings in `cfg`; the ones you usually change:

| Setting | Meaning |
|---|---|
| `paths.orthophoto` | Your orthophoto (default `data/orthophoto.tif`) |
| `features.block_size` | Block size in pixels (default 50) |
| `features.n_jobs` | CPU cores for the map (-1: all but one) |
| `validation.total_samples` | Number of validation samples (default 150) |

All outputs go to `outputs/`.

## Step 3: Create the annotation file, then annotate in QGIS

Step 3 writes `data/sampling_data.gpkg` with three layers:

- `points`: random seed points spread over the image (a 5 × 5 grid, 5 points per cell). They only suggest
  *where* to look, so the whole reach is covered.
- `annotations`: empty polygon layer with a `substrate` field — your training data.
- `roi`: optional polygon(s) limiting the work to an area of interest (e.g. the active channel).

In QGIS:

1. Load the orthophoto and the three layers of `sampling_data.gpkg`.
2. Near each seed point, draw a polygon over an area of **one** clearly recognisable substrate class and
   type its class in `substrate`. If the point falls on a transition, draw the polygon on a nearby
   homogeneous patch instead.
3. Add polygons for classes that are rare or missing, so that every class has **at least 5 polygons in
   different parts of the reach**. Several small polygons in different places are better than one large one.
4. Make polygons at least 3 × 3 blocks (about 3 × 3 m) and keep them away from class boundaries.
5. Optional: draw the area of interest in `roi`. To place the seed points only inside it, re-run Step 3 with
   `points_only=True` (your annotations are kept).
6. Save the layers.

Re-running Step 3 never overwrites an existing file; pass `overwrite=True` to start from scratch.

## Step 4: Extract features

Computes colour and texture features for every block that lies at least 90 % inside a polygon. It warns
about polygons too small to give any block and about possible typos in class names.

## Step 5: Train the model

Selects the useful features, evaluates the model with spatial cross-validation (whole polygons are held
out, so the scores reflect how well the model works on places it has not seen), and trains the final model.

Read the printed table: classes with low recall in **spatial** CV need more, or more varied, polygons.
A large gap between standard and spatial CV means the model memorises the annotated patches.
Step 9 explains what to do.

## Step 6: Classify the map

Writes to `outputs/maps`:

- `classified_map.tif`: one class per block (`class_dictionary.json` lists the codes);
- `classified_map_confidence.tif`: probability of the mapped class (%); low values mark uncertain areas.

Load both in QGIS and compare them with the orthophoto. The Piave reach (2 million blocks) takes about 4
minutes on a 14-core laptop.

## Step 7: Create the validation sample, then label it in QGIS

Step 7 draws a random sample of map blocks (more in large classes, at least 20 per class) and writes:

- `outputs/validation/validation_true_annotations.gpkg`: the blocks to label;
- `outputs/validation/validation_modeled.gpkg`: the map's answers (**do not open it while labelling**).

In QGIS, open `validation_true_annotations.gpkg` over the orthophoto and type the true class of each block
in `true_label` (class name, or its code from `class_dictionary.json`). Label what you see, not what you
expect the map to say.

Step 7 also prints how many samples give a ±2 % uncertainty of overall accuracy. With 20 samples a
class's accuracy is only known to about ±20 %; important rare classes need 50–100.

## Step 8: Accuracy and areas

Reads the labels and reports, with 95 % confidence intervals:

- **overall accuracy**;
- **user's accuracy**: of the area mapped as a class, the share that truly is that class;
- **producer's accuracy**: of the true area of a class, the share the map finds;
- **estimated area** of every class (ha), corrected for the errors found in the sample. Use these areas,
  rather than pixel counts of the map, for reporting.

Tables are saved in `outputs/validation` (`accuracy_summary.csv`, `accuracy_per_class.csv`,
`error_matrix_area.csv`).

After improving the model, you can assess the new map on the same labelled sample, without labelling
again: set `map_path` to the new map in Step 8.

## Step 9: Diagnostics — how to improve the model

Lists the most useful next steps, most important first, for example:

- a class with too few, too small or too clustered polygons;
- two classes the model confuses, with advice depending on why (wet/dry, grain size, bedrock vs. sediment,
  vegetation);
- a polygon whose label may be wrong;
- classes the map over- or under-estimates.

It also writes `outputs/diagnostics/suggested_annotations.gpkg`: places where the model is least sure.
Load it in QGIS; annotating some of these places (Step 3) and re-running Steps 4–6 usually helps most.
If they show something outside your class list (shadows, white water, roads), add a class or exclude it
with the ROI.

## Tips

- **Large orthophotos** read fastest as tiled GeoTIFFs:
  `gdal_translate -co TILED=YES -co COMPRESS=DEFLATE in.tif out.tif` (GDAL ships with QGIS).
- **Inside QGIS or when memory is short**, use `n_jobs = 1`.
- **In your own Python scripts** using `n_jobs > 1`, put the code under `if __name__ == '__main__':`.
- **Clear water only**: the bed must be visible. Exclude deep water, white water, glare and strong shadows
  with the ROI, or give them their own class.

## Troubleshooting

| Problem | Check |
|---|---|
| "No valid blocks found" | Polygons have a `substrate` value, overlap the orthophoto, and are larger than a block. |
| A class is never mapped | Step 9; add more and more varied polygons of that class. |
| Map empty inside the ROI | The `roi` polygons overlap the orthophoto and are in the same area. |
| Step 8 warns about unknown labels | Typos in `true_label`; names must match the training classes. |
