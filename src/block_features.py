import numpy as np
from affine import Affine
from rasterio.enums import ColorInterp
from rasterio.windows import Window
from skimage.color import rgb2lab

"""
Shared block feature core.

Training (features.py) and inference (inference.py) both go through
`read_rgb_window` + `features_for_blocks`, so a block produces identical
features whichever step computes it. Blocks live on one global grid anchored
at the orthophoto origin: block (r, c) covers pixels
[r*M:(r+1)*M, c*M:(c+1)*M], which is also pixel (r, c) of the classified map.
"""

# --- LBP Parameters ---
LBP_POINTS = 8
LBP_RADIUS = 1
LBP_METHOD = 'uniform'
LBP_N_BINS = LBP_POINTS + 2

# A block is only used (for training or prediction) when this fraction of its
# pixels holds valid image data. Slightly below 1 so that a few genuinely black
# pixels (deep shadow) do not invalidate a block, while mosaic-edge blocks with
# substantial padding are still excluded.
MIN_VALID_FRACTION = 0.99

# Texture channels: HSV value (max of R, G, B; 0-255) and CIELAB L* (0-100, truncated to uint8;
# kept as in the published workflow). 'lab_l8' is L* rescaled to 0-255.
TEXTURE_CHANNELS = ('hsv_v', 'lab_l')
GLCM_PROPS = ('contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation', 'entropy')

SPECTRAL_FEATURES = [
    'mean_r', 'mean_g', 'mean_b', 'mean_hsv_v', 'mean_lab_l',
    'norm_a_star', 'norm_b_star',
    'std_hsv_v', 'var_hsv_v', 'std_lab_l', 'var_lab_l',
    'c1_invariant', 'c2_invariant', 'c3_invariant',
]
GLCM_FEATURES = [f'glcm_{ch}_{p}' for ch in TEXTURE_CHANNELS for p in GLCM_PROPS]
LBP_FEATURES = [f'lbp_{ch}_bin_{i}' for ch in TEXTURE_CHANNELS for i in range(LBP_N_BINS)]

# Multi-scale texture (grain-size aware). Computed on HSV value and on L* rescaled to 0-255.
MS_CHANNELS = ('hsv_v', 'lab_l8')
GLCM_MS_LEVELS = 32
GLCM_MS_DISTANCES = (1, 2, 4)
GLCM_MS_ANGLES = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
GLCM_MS_PROPS = ('contrast', 'dissimilarity', 'homogeneity', 'ASM', 'correlation', 'entropy')
LBP_MS_RADII = (1, 2, 3)
LBP_MS_CHANNELS = ('hsv_v',)

GLCM_MS_FEATURES = [f'glcm_ms_{ch}_d{d}_{p}' for ch in MS_CHANNELS for d in GLCM_MS_DISTANCES for p in GLCM_MS_PROPS]
LBP_MS_FEATURES = [f'lbp_ms_{ch}_r{r}_bin_{i}' for ch in LBP_MS_CHANNELS for r in LBP_MS_RADII for i in range(LBP_N_BINS)]
GRADIENT_FEATURES = [f'grad_{ch}_{s}' for ch in MS_CHANNELS for s in ('mean', 'std')]

FEATURE_FAMILIES = {
    'spectral': SPECTRAL_FEATURES, 'glcm': GLCM_FEATURES, 'lbp': LBP_FEATURES,
    'glcm_ms': GLCM_MS_FEATURES, 'lbp_ms': LBP_MS_FEATURES, 'gradient': GRADIENT_FEATURES,
}
DEFAULT_FAMILIES = tuple(FEATURE_FAMILIES)          # all families; selection keeps the useful ones
ALL_FEATURES = [f for fam in FEATURE_FAMILIES.values() for f in fam]


def grid_shape(src, block_size):
    """(rows, cols) of complete blocks in the orthophoto; partial edge blocks are dropped."""
    return src.height // block_size, src.width // block_size


def grid_transform(src, block_size):
    """Affine transform of the block grid (one map pixel per block)."""
    return src.transform * Affine.scale(block_size, block_size)


def read_rgb_window(src, row0, col0, n_rows, n_cols, block_size):
    """
    Reads a block-aligned window: blocks [row0:row0+n_rows, col0:col0+n_cols].

    Returns
    -------
    rgb : (H, W, 3) uint8
    valid : (H, W) bool -- alpha band (or nodata mask) AND not a pure black/white fill pixel.
    """
    win = Window(col0 * block_size, row0 * block_size, n_cols * block_size, n_rows * block_size)
    rgb = np.transpose(src.read((1, 2, 3), window=win), (1, 2, 0))
    if rgb.dtype != np.uint8:
        raise TypeError(f"Expected an 8-bit RGB orthophoto, got {rgb.dtype}.")
    if ColorInterp.alpha in src.colorinterp:
        valid = src.read(src.colorinterp.index(ColorInterp.alpha) + 1, window=win) > 0
    else:
        valid = src.dataset_mask(window=win) > 0
    valid &= is_image_pixel(rgb, axis=2)
    return rgb, valid


def is_image_pixel(rgb, axis):
    """
    False for mosaic fill pixels: pure black (0,0,0) or pure white (255,255,255).
    Orthophotos without alpha band or nodata value use either colour outside the mosaic.
    """
    return rgb.any(axis=axis) & ~(rgb == 255).all(axis=axis)


def valid_overview(src, max_dim=2000):
    """
    Low-resolution validity mask of the whole orthophoto (same rule as `read_rgb_window`).

    Returns (mask, transform), with the longest side at most `max_dim` pixels.
    Uses the file's overviews when present, so it stays fast on large mosaics.
    """
    scale = max(src.height, src.width) / max_dim
    out_h, out_w = max(1, int(src.height / scale)), max(1, int(src.width / scale))
    rgb = src.read((1, 2, 3), out_shape=(3, out_h, out_w))
    valid = is_image_pixel(rgb, axis=0)
    if ColorInterp.alpha in src.colorinterp:
        valid &= src.read(src.colorinterp.index(ColorInterp.alpha) + 1, out_shape=(out_h, out_w)) > 0
    transform = src.transform * Affine.scale(src.width / out_w, src.height / out_h)
    return valid, transform


def block_fraction(mask, block_size):
    """Fraction of True pixels per block, for a block-aligned (H, W) mask."""
    h, w = mask.shape
    m = mask.reshape(h // block_size, block_size, w // block_size, block_size)
    return m.mean(axis=(1, 3))


def _block_stack(arr, block_size, rows, cols):
    """The selected blocks of a block-aligned (H, W, ...) array, as an (n, M, M, ...) array."""
    M = block_size
    H, W = arr.shape[:2]
    v = arr.reshape(H // M, M, W // M, M, *arr.shape[2:]).swapaxes(1, 2)
    return v[rows, cols]


def _pair_offset(distance, angle):
    """Pixel offset (rows, cols) of a GLCM direction, using skimage's convention."""
    return int(np.round(np.sin(angle) * distance)), int(np.round(np.cos(angle) * distance))


def _pairs(q, dr, dc):
    """Ordered pixel pairs (p, p + (dr, dc)) inside every block of q (n, M, M), one row per block."""
    n, M, _ = q.shape
    r0, r1 = max(0, -dr), M - max(0, dr)
    c0, c1 = max(0, -dc), M - max(0, dc)
    a = q[:, r0:r1, c0:c1].reshape(n, -1)
    b = q[:, r0 + dr:r1 + dr, c0 + dc:c1 + dc].reshape(n, -1)
    return a, b


def glcm_properties(q, levels, dr, dc, entropy_eps):
    """
    Properties of the symmetric, normalised GLCM of every block in q (n, M, M) for one offset.
    Values match skimage's graycomatrix/graycoprops (symmetric=True, normed=True).

    Few grey levels: the co-occurrence matrix of every block is counted with one bincount and
    all properties are weighted sums over it. Many grey levels (256): contrast, dissimilarity,
    homogeneity and correlation are computed from the pixel pairs directly; ASM and entropy
    from the run lengths of the sorted pair codes (levels <= 256).
    """
    a, b = _pairs(q.astype(np.int64), dr, dc)
    n, m = a.shape
    if levels <= 64:
        codes = (np.arange(n)[:, None] * levels + a) * levels + b
        C = np.bincount(codes.ravel(), minlength=n * levels ** 2).reshape(n, levels, levels)
        P = (C + C.transpose(0, 2, 1)).reshape(n, -1) / (2.0 * m)
        i, j = np.divmod(np.arange(levels ** 2), levels)
        d2 = (i - j) ** 2
        mu = P @ i
        var = P @ (i ** 2) - mu ** 2
        cov = P @ (i * j) - mu ** 2
        asm = (P ** 2).sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            ent = -(P * np.log2(P + entropy_eps)).sum(axis=1)
            corr = cov / var
        return {
            'contrast': P @ d2, 'dissimilarity': P @ np.abs(i - j), 'homogeneity': P @ (1.0 / (1.0 + d2)),
            'correlation': np.where(var > 1e-12, corr, 1.0),
            'ASM': asm, 'energy': np.sqrt(asm), 'entropy': ent,
        }

    d = (a - b).astype(float)                   # symmetric GLCM: (i, j) and (j, i) weigh the same
    out = {
        'contrast': (d ** 2).mean(axis=1),
        'dissimilarity': np.abs(d).mean(axis=1),
        'homogeneity': (1.0 / (1.0 + d ** 2)).mean(axis=1),
    }
    mu = (a.mean(axis=1) + b.mean(axis=1)) / 2
    ac, bc = a - mu[:, None], b - mu[:, None]
    var = ((ac ** 2).mean(axis=1) + (bc ** 2).mean(axis=1)) / 2
    with np.errstate(invalid='ignore', divide='ignore'):
        corr = (ac * bc).mean(axis=1) / var
    out['correlation'] = np.where(var > 1e-12, corr, 1.0)

    # Pair probabilities from run lengths of the sorted pair codes of every block
    codes = np.concatenate([a * levels + b, b * levels + a], axis=1).astype(np.uint16)
    flat = np.sort(codes, axis=1, kind='stable').ravel()
    new_run = np.empty(flat.size, dtype=bool)
    new_run[0] = True
    new_run[1:] = flat[1:] != flat[:-1]
    new_run[::2 * m] = True                     # runs never cross blocks
    starts = np.flatnonzero(new_run)
    p = np.diff(np.append(starts, flat.size)) / (2.0 * m)
    blk = starts // (2 * m)
    asm = np.bincount(blk, p ** 2, n)
    ent = np.bincount(blk, -p * np.log2(p + entropy_eps), n)
    out['ASM'], out['energy'], out['entropy'] = asm, np.sqrt(asm), ent
    return out


def lbp_uniform(blocks, radius, points=LBP_POINTS):
    """
    Rotation-invariant uniform LBP codes of every block in blocks (n, M, M), following skimage's
    local_binary_pattern(method='uniform'): neighbours on a circle by bilinear interpolation,
    zero outside the block. Neighbours within 1e-9 of the centre count as ties (>= centre), so
    results do not depend on floating-point rounding of the interpolation (skimage can differ
    from this in such exact ties).
    """
    n, M, _ = blocks.shape
    img = blocks.astype(np.float64)
    pad = int(np.ceil(radius)) + 1
    padded = np.zeros((n, M + 2 * pad, M + 2 * pad))
    padded[:, pad:pad + M, pad:pad + M] = img
    window = lambda dr, dc: padded[:, pad + dr:pad + dr + M, pad + dc:pad + dc + M]

    angles = 2 * np.pi * np.arange(points, dtype=np.float64) / points
    signs = np.empty((points, n, M, M), dtype=np.int8)
    for k, (rp, cp) in enumerate(zip(np.round(-radius * np.sin(angles), 5), np.round(radius * np.cos(angles), 5))):
        r0, c0, r1, c1 = int(np.floor(rp)), int(np.floor(cp)), int(np.ceil(rp)), int(np.ceil(cp))
        fr, fc = rp - r0, cp - c0
        if fr == 0 and fc == 0:                           # neighbour on the pixel grid
            signs[k] = window(r0, c0) >= img
            continue
        top = (1 - fc) * window(r0, c0) + fc * window(r0, c1)
        bottom = (1 - fc) * window(r1, c0) + fc * window(r1, c1)
        signs[k] = ((1 - fr) * top + fr * bottom - img) >= -1e-9

    changes = np.abs(np.diff(signs, axis=0)).sum(axis=0)  # transitions between consecutive neighbours
    return np.where(changes <= 2, signs.sum(axis=0), points + 1)


def _lbp_histograms(blocks, radius):
    """Uniform LBP histogram (LBP_N_BINS bins, density) of every block in blocks (n, M, M)."""
    codes = lbp_uniform(blocks, radius)
    n, M, _ = blocks.shape
    counts = np.bincount((np.arange(n)[:, None] * LBP_N_BINS + codes.reshape(n, -1)).ravel(),
                         minlength=n * LBP_N_BINS).reshape(n, LBP_N_BINS)
    return counts / (M * M)


def _needed(names):
    """Works out which feature groups must be computed to produce `names`."""
    names = set(names)
    has = lambda prefix: any(n.startswith(prefix) for n in names)
    return {
        'spectral': bool(names & set(SPECTRAL_FEATURES)),
        'glcm': {ch for ch in TEXTURE_CHANNELS if has(f'glcm_{ch}_')},
        'lbp': {ch for ch in TEXTURE_CHANNELS if has(f'lbp_{ch}_')},
        'glcm_ms': {(ch, d) for ch in MS_CHANNELS for d in GLCM_MS_DISTANCES if has(f'glcm_ms_{ch}_d{d}_')},
        'lbp_ms': {(ch, r) for ch in LBP_MS_CHANNELS for r in LBP_MS_RADII if has(f'lbp_ms_{ch}_r{r}_')},
        'gradient': {ch for ch in MS_CHANNELS if has(f'grad_{ch}_')},
    }


def compute_features(rgb, lab, channels, needed):
    """
    Features of a stack of blocks.

    rgb (n, M, M, 3) uint8, lab (n, M, M, 3) float, channels: name -> (n, M, M) uint8.
    Returns a dict feature name -> (n,) array.
    """
    f = {}
    if needed['spectral']:
        mean_rgb = rgb.mean(axis=(1, 2))
        mean_lab = lab.mean(axis=(1, 2))
        v, l_star = channels['hsv_v'], lab[..., 0]
        f.update({
            'mean_r': mean_rgb[:, 0], 'mean_g': mean_rgb[:, 1], 'mean_b': mean_rgb[:, 2],
            'mean_hsv_v': v.mean(axis=(1, 2)), 'mean_lab_l': l_star.mean(axis=(1, 2)),
            'norm_a_star': mean_lab[:, 1] / (mean_lab[:, 0] + 1e-6),
            'norm_b_star': mean_lab[:, 2] / (mean_lab[:, 0] + 1e-6),
            'std_hsv_v': v.std(axis=(1, 2)), 'var_hsv_v': v.var(axis=(1, 2)),
            'std_lab_l': l_star.std(axis=(1, 2)), 'var_lab_l': l_star.var(axis=(1, 2)),
        })
        R, G, B = (rgb[..., k].astype(float) for k in range(3))
        f['c1_invariant'] = np.arctan(R / (np.maximum(G, B) + 1e-6)).mean(axis=(1, 2))
        f['c2_invariant'] = np.arctan(G / (np.maximum(R, B) + 1e-6)).mean(axis=(1, 2))
        f['c3_invariant'] = np.arctan(B / (np.maximum(R, G) + 1e-6)).mean(axis=(1, 2))

    for ch in needed['glcm']:
        props = glcm_properties(channels[ch], 256, 0, 1, entropy_eps=1e-6)
        f.update({f'glcm_{ch}_{p}': props[p] for p in GLCM_PROPS})

    quantised = {}
    for ch, d in sorted(needed['glcm_ms']):
        if ch not in quantised:
            quantised[ch] = (channels[ch].astype(np.int32) * GLCM_MS_LEVELS // 256).astype(np.uint8)
        per_angle = [glcm_properties(quantised[ch], GLCM_MS_LEVELS, *_pair_offset(d, a), entropy_eps=1e-12)
                     for a in GLCM_MS_ANGLES]
        for p in GLCM_MS_PROPS:
            f[f'glcm_ms_{ch}_d{d}_{p}'] = np.mean([pa[p] for pa in per_angle], axis=0)

    for ch in needed['lbp']:
        h = _lbp_histograms(channels[ch], LBP_RADIUS)
        f.update({f'lbp_{ch}_bin_{i}': h[:, i] for i in range(LBP_N_BINS)})
    for ch, r in sorted(needed['lbp_ms']):
        h = _lbp_histograms(channels[ch], r)
        f.update({f'lbp_ms_{ch}_r{r}_bin_{i}': h[:, i] for i in range(LBP_N_BINS)})

    for ch in needed['gradient']:
        gy, gx = np.gradient(channels[ch].astype(float), axis=(1, 2))
        mag = np.hypot(gx, gy)
        f[f'grad_{ch}_mean'], f[f'grad_{ch}_std'] = mag.mean(axis=(1, 2)), mag.std(axis=(1, 2))
    return f


def features_for_blocks(rgb, block_size, block_ok, feature_names):
    """
    Computes features for every selected block of a block-aligned RGB window.

    Parameters
    ----------
    rgb : (H, W, 3) uint8, H and W multiples of block_size.
    block_ok : (H/M, W/M) bool, blocks to compute.
    feature_names : ordered list of features to return.

    Returns
    -------
    X : (n_ok, len(feature_names)) float array, rows in np.nonzero(block_ok) order.
    """
    rows, cols = np.nonzero(block_ok)
    X = np.zeros((len(rows), len(feature_names)), dtype=float)
    if len(rows) == 0:
        return X

    needed = _needed(feature_names)
    lab = rgb2lab(rgb)
    channels = {
        'hsv_v': rgb.max(axis=2),
        'lab_l': lab[:, :, 0].astype(np.uint8),
        'lab_l8': np.clip(np.round(lab[:, :, 0] * 2.55), 0, 255).astype(np.uint8),
    }
    stack = lambda arr: _block_stack(arr, block_size, rows, cols)
    feats = compute_features(stack(rgb), stack(lab), {k: stack(v) for k, v in channels.items()}, needed)
    X = np.column_stack([feats[n] for n in feature_names]).astype(float)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
