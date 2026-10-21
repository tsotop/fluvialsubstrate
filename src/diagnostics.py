import os
import json
from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd

"""
Model diagnostics: a short, prioritised list of tips on how to improve the classification.

Each step of the workflow leaves results behind; `diagnose` reads whatever is available
(annotations/feature table, trained model, map, validation) and turns it into tips with a
severity, the classes concerned, what was found and what to do. The tips are plain data
(`Tip`), so they can be printed in a notebook, written to Markdown/JSON, or shown in a GUI.
"""

# --- Thresholds ---
MIN_POLYGONS = 5            # polygons per class recommended (spatial CV needs at least 3)
MIN_BLOCKS = 50             # training blocks per class
DOMINANT_POLYGON = 0.5      # one polygon holding more than this share of its class's blocks
CLUSTERED_SPAN = 0.15       # class polygons spread over less than this share of the annotated reach
MAX_IMBALANCE = 10          # class weights are capped at this ratio (see ml_utils.Weighting)
LOW_RECALL = 0.6            # spatial-CV recall (or validation accuracy) considered poor
CONFUSION_SHARE = 0.2       # share of a class predicted as another class worth reporting
POLYGON_AGREEMENT = 0.3     # polygons agreeing less than this with the model: check label
OPTIMISM_GAP = 0.25         # standard minus spatial CV macro-F1
LOW_CONFIDENCE = 0.5        # probability of the mapped class considered uncertain
WIDE_CI = 0.2               # half-width of a 95 % CI considered too wide

SEVERITIES = ('high', 'medium', 'info')
STAGES = ('annotation', 'training', 'map', 'validation')
SIZE_WORDS = {'small', 'large', 'fine', 'coarse', 'medium', 'sand', 'gravel', 'pebble', 'cobble', 'boulder',
              'silt', 'clay'}
STRUCTURE_WORDS = {'bedrock', 'rock', 'concrete', 'riprap'}
MOISTURE_WORDS = {'wet', 'dry', 'submerged', 'emerged'}
VEGETATION_WORDS = {'veg', 'vegetation', 'grass', 'shrub', 'tree', 'algae'}


@dataclass
class Tip:
    severity: str           # 'high', 'medium' or 'info'
    stage: str              # 'annotation', 'training', 'map' or 'validation'
    message: str            # what was found
    action: str             # what to do about it
    classes: list = field(default_factory=list)

    def sort_key(self):
        return SEVERITIES.index(self.severity), STAGES.index(self.stage)


def _words(name):
    return set(str(name).lower().replace('_', '-').replace(' ', '-').split('-'))


def _pair_advice(a, b, block_m):
    """Advice for two confused classes, depending on how their names differ."""
    wa, wb = _words(a), _words(b)
    diff = wa ^ wb
    if (wa | wb) & VEGETATION_WORDS and diff & VEGETATION_WORDS:
        return ("Vegetation is confused with substrate, typically where vegetation is sparse, partly "
                "covers the bed, or casts shadows.",
                "Decide whether sparse vegetation is labelled as vegetation or as the substrate below and "
                "apply it consistently; keep polygons away from vegetation edges.")
    moisture, size, structure = diff & MOISTURE_WORDS, diff & SIZE_WORDS, diff & STRUCTURE_WORDS
    if diff and diff <= MOISTURE_WORDS:
        return ("The two classes differ only in moisture; errors usually sit at the water line or on damp, "
                "partly wet surfaces.",
                "Keep polygons clearly on one side of the water line and label damp margins consistently "
                "(or leave them out).")
    if structure:
        return ("Bedrock and loose sediment are told apart by structure (continuous surface vs. individual "
                "grains), which water, algae or shadows can hide.",
                "Annotate bedrock and sediment where the difference is clearly visible, also under water; "
                "avoid polygons on transitions.")
    if diff and diff <= SIZE_WORDS | MOISTURE_WORDS:
        size_text = (f"different grain sizes, separated mainly by texture, which requires grains to be several "
                     f"pixels across (blocks are {block_m:.2f} m)")
        if moisture:
            return ("They differ in moisture and grain size: wet surfaces look darker and smoother, so a wet, "
                    "coarse surface can resemble a dry, finer one.",
                    "Annotate both classes on both sides of the water line where they occur, so the model sees "
                    "wet and dry examples of each grain size.")
        return (f"These are {size_text}.",
                "Annotate clear, homogeneous examples of both; if they keep overlapping, check that the smaller "
                "grains are resolved in the orthophoto, or consider merging the two classes.")
    return ("The classes look alike to the model.",
            "Check that they are visually distinct in the orthophoto and labelled consistently; add examples "
            "of both where they occur side by side.")


# --------------------------------------------------------------------------- annotation
def annotation_tips(feature_meta, block_counts=None):
    """Tips from the annotation polygons (feature table metadata written by run_extraction)."""
    tips = []
    polys = pd.DataFrame(feature_meta.get('polygons', []))
    if polys.empty:
        return tips
    block_m = float(np.sqrt(polys['block_area_m2'].iloc[0]))
    if block_counts is None:
        block_counts = polys.groupby('class')['blocks_used'].sum()

    # Position of every polygon along the main axis of all annotations (rivers are elongated)
    xy = polys[['x', 'y']].to_numpy()
    xy_c = xy - xy.mean(axis=0)
    axis = np.linalg.svd(xy_c, full_matrices=False)[2][0] if len(polys) > 1 else np.array([1.0, 0.0])
    polys['along'] = xy_c @ axis
    reach = polys['along'].max() - polys['along'].min()

    for cls, p in polys.groupby('class'):
        used = p[p['blocks_used'] > 0]
        n_blocks = int(block_counts.get(cls, 0))
        if len(used) < 3:
            tips.append(Tip('high', 'annotation',
                            f"'{cls}' has only {len(used)} usable polygon(s): spatial cross-validation cannot test it "
                            f"properly and the model sees too few independent examples.",
                            f"Draw at least {MIN_POLYGONS} polygons of '{cls}' in different parts of the reach.", [cls]))
        elif len(used) < MIN_POLYGONS:
            tips.append(Tip('medium', 'annotation',
                            f"'{cls}' has {len(used)} usable polygons.",
                            f"Aim for at least {MIN_POLYGONS} polygons per class, spread over the reach.", [cls]))

        touched, kept = p['blocks_touched'].sum(), p['blocks_used'].sum()
        if n_blocks < MIN_BLOCKS:
            msg = f"'{cls}' has only {n_blocks} training blocks."
            if touched > 0 and kept / touched < 0.5:
                msg += (f" Its polygons touch {touched} blocks but only {kept} lie mostly inside them: the polygons "
                        f"are small or narrow for {block_m:.2f} m blocks.")
            tips.append(Tip('high' if n_blocks < MIN_BLOCKS / 2 else 'medium', 'annotation', msg,
                            f"Draw larger or wider polygons of '{cls}' (at least about {3 * block_m:.1f} x "
                            f"{3 * block_m:.1f} m), or use a smaller block size.", [cls]))

        empty = p[p['blocks_used'] == 0]['label_id'].tolist()
        if empty:
            tips.append(Tip('medium', 'annotation',
                            f"{len(empty)} '{cls}' polygon(s) give no training block (smaller or narrower than a "
                            f"{block_m:.2f} m block): label_id {empty}.",
                            "Enlarge or remove them.", [cls]))

        if len(used) > 1 and kept > 0:
            top = used.loc[used['blocks_used'].idxmax()]
            share = top['blocks_used'] / kept
            if share > DOMINANT_POLYGON:
                tips.append(Tip('medium', 'annotation',
                                f"Polygon {int(top['label_id'])} holds {share:.0%} of the '{cls}' blocks: the model "
                                f"mostly learns that single patch.",
                                f"Add '{cls}' polygons elsewhere; several smaller polygons from different places "
                                f"generalise better than one large one.", [cls]))

        span = p['along'].max() - p['along'].min()
        if len(p) >= 2 and reach > 0 and span / reach < CLUSTERED_SPAN:
            tips.append(Tip('medium', 'annotation',
                            f"All '{cls}' polygons lie within about {span:.0f} m of each other along a reach annotated "
                            f"over {reach:.0f} m.",
                            f"If '{cls}' occurs elsewhere in the reach, annotate it there too, so the model learns "
                            f"its variability (light, moisture, sorting).", [cls]))

    counts = pd.Series(block_counts)
    counts = counts[counts > 0]
    if len(counts) > 1 and counts.max() / counts.min() > MAX_IMBALANCE:
        rare = counts[counts.max() / counts > MAX_IMBALANCE].index.tolist()
        tips.append(Tip('info', 'annotation',
                        f"{', '.join(map(repr, rare))} have more than {MAX_IMBALANCE}x fewer training blocks than "
                        f"'{counts.idxmax()}'; class weights are capped at {MAX_IMBALANCE}x, so they stay "
                        f"under-represented.",
                        "Annotate more of these classes where they occur.", rare))
    return tips


# --------------------------------------------------------------------------- training
def training_tips(cv_results, block_m=None):
    """Tips from cross-validation results (cv_results.json written by train_pipeline)."""
    tips = []
    spatial = cv_results.get('spatial_cv')
    if not spatial or 'confusion' not in spatial:
        return tips
    classes = cv_results['classes']
    C = np.asarray(spatial['confusion'], dtype=float)           # rows: true, columns: predicted
    R = C / np.maximum(C.sum(axis=1, keepdims=True), 1)
    report = spatial['report']

    for i, cls in enumerate(classes):
        recall = report[cls]['recall']
        if C[i].sum() and recall < LOW_RECALL:
            others = [(classes[j], R[i, j]) for j in np.argsort(-R[i]) if j != i and R[i, j] >= 0.1][:2]
            where = "; mostly predicted as " + ", ".join(f"'{c}' ({s:.0%})" for c, s in others) if others else ""
            tips.append(Tip('high', 'training',
                            f"'{cls}' is recognised in only {recall:.0%} of its blocks when tested on polygons the "
                            f"model has not seen{where}.",
                            f"Add '{cls}' polygons in new places: more independent examples help most.", [cls]))

    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            share = max(R[i, j], R[j, i])
            if share >= CONFUSION_SHARE:
                why, action = _pair_advice(classes[i], classes[j], block_m or float('nan'))
                tips.append(Tip('high' if share >= 0.35 else 'medium', 'training',
                                f"'{classes[i]}' and '{classes[j]}' are confused ({R[i, j]:.0%} of '{classes[i]}' "
                                f"predicted as '{classes[j]}', {R[j, i]:.0%} the other way). {why}",
                                action, [classes[i], classes[j]]))

    polygons = spatial.get('polygons', [])
    class_blocks = pd.DataFrame(polygons).groupby('class')['n_blocks'].sum() if polygons else {}
    # A polygon holding most of its class cannot be judged: without it the model hardly knows the class
    suspects = sorted((p for p in polygons
                       if p['agreement'] < POLYGON_AGREEMENT and p['n_blocks'] >= 5
                       and p['n_blocks'] < 0.5 * class_blocks[p['class']]
                       and report[p['class']]['recall'] >= 0.5), key=lambda p: p['agreement'])
    for p in suspects[:3]:
        tips.append(Tip('medium', 'training',
                        f"Polygon {p['label_id']} ('{p['class']}', {p['n_blocks']} blocks) is predicted as "
                        f"'{p['most_predicted']}' in {p['most_predicted_share']:.0%} of its blocks, although "
                        f"'{p['class']}' is otherwise recognised well.",
                        f"Check its label and that it covers only '{p['class']}'. If it is correct, '{p['class']}' "
                        f"looks like '{p['most_predicted']}' there: add similar examples.", [p['class']]))

    standard = cv_results.get('standard_cv')
    if standard:
        gap = standard['macro_f1'] - spatial['macro_f1']
        if gap > OPTIMISM_GAP:
            tips.append(Tip('medium', 'training',
                            f"Macro-F1 is {standard['macro_f1']:.2f} on random blocks but {spatial['macro_f1']:.2f} on "
                            f"unseen polygons: the model recognises the annotated patches much better than new places.",
                            "More, smaller polygons spread over the reach generalise better than a few large ones.",
                            []))
    return tips


# --------------------------------------------------------------------------- map
def map_tips(map_path, class_names, output_dir=None, n_suggestions=15):
    """Tips from the classified map and its confidence raster; writes suggested annotation points."""
    import rasterio
    import geopandas as gpd
    from scipy.ndimage import uniform_filter, maximum_filter

    tips = []
    conf_path = os.path.splitext(map_path)[0] + '_confidence.tif'
    with rasterio.open(map_path) as m:
        labels, transform, crs, nodata = m.read(1), m.transform, m.crs, m.nodata
    if not os.path.exists(conf_path):
        return tips
    with rasterio.open(conf_path) as c:
        conf = c.read(1).astype(float) / 100
    valid = labels != nodata
    low = valid & (conf < LOW_CONFIDENCE)
    if not valid.any():
        return tips

    share_low = low.sum() / valid.sum()
    if share_low > 0.25:
        tips.append(Tip('medium', 'map',
                        f"{share_low:.0%} of the map has less than {LOW_CONFIDENCE:.0%} confidence in the mapped class.",
                        "The training data do not cover these areas well; see the suggested annotation locations.", []))
    for k, cls in sorted(class_names.items()):
        n = (labels == k).sum()
        if n == 0:
            tips.append(Tip('high', 'map', f"'{cls}' is never mapped, although it was trained.",
                            f"The model always prefers another class: add more, and more varied, '{cls}' polygons.",
                            [cls]))
        elif (low & (labels == k)).sum() / n > 0.5:
            tips.append(Tip('info', 'map',
                            f"Most of the area mapped as '{cls}' is uncertain (confidence < {LOW_CONFIDENCE:.0%}).",
                            f"Treat '{cls}' areas with care; check them against the orthophoto.", [cls]))

    # Suggested annotation locations: centres of the largest low-confidence patches, spread out
    if output_dir is not None and low.any():
        density = uniform_filter(low.astype(float), size=5)
        sep = max(10, int(max(labels.shape) / 25))
        peaks = (density == maximum_filter(density, size=sep)) & low & (density > 0.5)
        r, c = np.nonzero(peaks)
        order = np.argsort(-density[r, c])[:n_suggestions]
        pts = []
        for i in order:
            x, y = rasterio.transform.xy(transform, r[i], c[i], offset='center')
            pts.append({'geometry': gpd.points_from_xy([x], [y])[0], 'mapped_class': class_names.get(int(labels[r[i], c[i]])),
                        'confidence': float(conf[r[i], c[i]]), 'uncertain_share_nearby': float(density[r[i], c[i]])})
        if pts:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, 'suggested_annotations.gpkg')
            gpd.GeoDataFrame(pts, crs=crs).to_file(path, driver='GPKG')
            tips.append(Tip('info', 'map',
                            f"{len(pts)} locations where the model is least confident were written to {path}.",
                            "Look at them in QGIS: annotating some (with the correct class) helps the model most. If "
                            "they show something outside your class list (shadow, white water, roads, buildings), add a "
                            "class for it or exclude it with the ROI.", []))
    return tips


# --------------------------------------------------------------------------- validation
def validation_tips(validation_dir):
    """Tips from the design-based validation tables written by validation.calculate_accuracy."""
    tips = []
    per_class_path = os.path.join(validation_dir, 'accuracy_per_class.csv')
    matrix_path = os.path.join(validation_dir, 'error_matrix_area.csv')
    if not os.path.exists(per_class_path):
        return tips
    pc = pd.read_csv(per_class_path)
    em = pd.read_csv(matrix_path, index_col=0) if os.path.exists(matrix_path) else None

    for _, r in pc.iterrows():
        cls = r['class']
        if r['n_mapped'] >= 5 and r['user_accuracy'] < LOW_RECALL:
            mostly = ''
            if em is not None and cls in em.index:
                row = em.loc[cls].drop(cls, errors='ignore')
                if row.max() > 0:
                    mostly = f"; most of the rest is '{row.idxmax()}'"
            tips.append(Tip('high', 'validation',
                            f"Of the area mapped as '{cls}', only {r['user_accuracy']:.0%} is '{cls}'{mostly}.",
                            f"The model over-predicts '{cls}': add polygons of the classes it is mistaken for, "
                            f"especially where they look like '{cls}'.", [cls]))
        if r['n_reference'] >= 5 and r['producer_accuracy'] < LOW_RECALL:
            mostly = ''
            if em is not None and cls in em.columns:
                col = em[cls].drop(cls, errors='ignore')
                if col.max() > 0:
                    mostly = f"; it is mostly mapped as '{col.idxmax()}'"
            tips.append(Tip('high', 'validation',
                            f"Only {r['producer_accuracy']:.0%} of the true '{cls}' area is mapped as '{cls}'{mostly}.",
                            f"The model misses '{cls}': add more '{cls}' polygons, especially where it resembles the "
                            f"class it is mapped as.", [cls]))
        half = 1.96 * r['user_accuracy_se']
        if r['n_mapped'] > 0 and np.isfinite(half) and half > WIDE_CI:
            needed = int(np.ceil(r['user_accuracy'] * (1 - r['user_accuracy']) / (0.1 / 1.96) ** 2))
            tips.append(Tip('info', 'validation',
                            f"User's accuracy of '{cls}' is uncertain (±{half:.0%}, {int(r['n_mapped'])} samples).",
                            f"About {needed} validation samples of '{cls}' would give ±10%.", [cls]))
        if 'area_ha' in r and r.get('mapped_area_ha', 0) > 0 and np.isfinite(r['area_ha']):
            ratio = r['area_ha'] / r['mapped_area_ha']
            if ratio > 1.5 or ratio < 1 / 1.5:
                tips.append(Tip('info', 'validation',
                                f"The map {'under' if ratio > 1 else 'over'}-estimates the area of '{cls}': mapped "
                                f"{r['mapped_area_ha']:.2f} ha, estimated {r['area_ha']:.2f} ± {r['area_ha_ci']:.2f} ha.",
                                "Use the estimated (error-adjusted) areas for reporting.", [cls]))
    return tips


# --------------------------------------------------------------------------- report
def diagnose(features_csv=None, model_dir=None, map_path=None, validation_dir=None, output_dir=None, show=True):
    """
    Collects tips from every available stage and returns them sorted by severity.

    features_csv : feature table written by run_extraction (its *_meta.json holds polygon statistics).
    model_dir : directory with cv_results.json (train_pipeline with evaluate=True).
    map_path : classified map (its *_confidence.tif is read next to it).
    validation_dir : directory with the tables written by calculate_accuracy.
    output_dir : where diagnostics.md, diagnostics.json and suggested_annotations.gpkg are written.
    """
    tips, block_m, class_names = [], None, None
    if features_csv is not None:
        from .features import feature_meta_path
        meta_path = feature_meta_path(features_csv)
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            counts = pd.read_csv(features_csv, usecols=['class'])['class'].value_counts()
            tips += annotation_tips(meta, counts)
            if meta.get('polygons'):
                block_m = float(np.sqrt(meta['polygons'][0]['block_area_m2']))
    if model_dir is not None and os.path.exists(os.path.join(model_dir, 'cv_results.json')):
        with open(os.path.join(model_dir, 'cv_results.json')) as f:
            cv = json.load(f)
        tips += training_tips(cv, block_m)
        class_names = dict(enumerate(cv.get('classes', [])))
    if map_path is not None:
        dict_path = os.path.join(os.path.dirname(map_path), 'class_dictionary.json')
        if os.path.exists(dict_path):
            with open(dict_path) as f:
                class_names = {int(k): v for k, v in json.load(f).items()}
        if class_names:
            tips += map_tips(map_path, class_names, output_dir)
    if validation_dir is not None:
        tips += validation_tips(validation_dir)

    tips.sort(key=Tip.sort_key)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'diagnostics.md'), 'w') as f:
            f.write(format_report(tips))
        with open(os.path.join(output_dir, 'diagnostics.json'), 'w') as f:
            json.dump([asdict(t) for t in tips], f, indent=2)
    if show:
        print(format_report(tips))
    return tips


def format_report(tips, top=5):
    """Markdown report: the `top` most important tips first, then all tips by workflow stage."""
    if not tips:
        return "# Model diagnostics\n\nNo issues found.\n"
    icons = {'high': '🔴', 'medium': '🟠', 'info': '🔵'}
    counts = {s: sum(t.severity == s for t in tips) for s in SEVERITIES}
    item = lambda t: [f"- {icons[t.severity]} {t.message}  ", f"  → {t.action}"]
    lines = ["# Model diagnostics", "",
             f"{counts['high']} important, {counts['medium']} worth checking, {counts['info']} for information.", "",
             "## Start here", ""]
    for t in tips[:top]:
        lines += item(t)
    for stage in STAGES:
        rest = [t for t in tips[top:] if t.stage == stage]
        if rest:
            lines += ["", f"## {stage.capitalize()}", ""]
            for t in rest:
                lines += item(t)
    lines.append("")
    return "\n".join(lines)
