"""
Per-block reference implementation of the block features (skimage-based, one block at a time).

The library (src/block_features.py) computes the same features vectorised over many blocks;
tests/test_feature_equivalence.py checks that both give identical values.
"""
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from src.block_features import (
    GLCM_PROPS, GLCM_MS_LEVELS, GLCM_MS_DISTANCES, GLCM_MS_ANGLES, GLCM_MS_PROPS, LBP_MS_RADII,
    LBP_POINTS, LBP_RADIUS, LBP_METHOD, LBP_N_BINS, TEXTURE_CHANNELS, MS_CHANNELS, LBP_MS_CHANNELS,
    SPECTRAL_FEATURES,
)


def get_color_invariants(rgb_patch):
    """Calculates the mean c1, c2, c3 color invariant features for a patch."""
    epsilon = 1e-6
    R, G, B = rgb_patch[:, :, 0].astype(float), rgb_patch[:, :, 1].astype(float), rgb_patch[:, :, 2].astype(float)
    c1 = np.mean(np.arctan(R / (np.maximum(G, B) + epsilon)))
    c2 = np.mean(np.arctan(G / (np.maximum(R, B) + epsilon)))
    c3 = np.mean(np.arctan(B / (np.maximum(R, G) + epsilon)))
    return {'c1_invariant': c1, 'c2_invariant': c2, 'c3_invariant': c3}


def get_glcm_features(gray_patch_int, prefix):
    """Calculates all GLCM properties for a given 8-bit grayscale patch."""
    max_val = int(np.max(gray_patch_int))
    levels = max(max_val + 1, 2)
    glcm = graycomatrix(gray_patch_int, distances=[1], angles=[0], levels=levels, symmetric=True, normed=True)
    feats = {f'glcm_{prefix}_{p}': graycoprops(glcm, p)[0, 0] for p in GLCM_PROPS if p != 'entropy'}
    feats[f'glcm_{prefix}_entropy'] = -np.sum(glcm * np.log2(glcm + 1e-6))
    return feats


def get_lbp_histogram(gray_patch_int, prefix):
    """Calculates the uniform LBP histogram for a given 8-bit grayscale patch."""
    lbp = local_binary_pattern(gray_patch_int, LBP_POINTS, LBP_RADIUS, LBP_METHOD)
    hist, _ = np.histogram(lbp.ravel(), density=True, bins=np.arange(0, LBP_N_BINS + 1))
    return {f'lbp_{prefix}_bin_{i}': val for i, val in enumerate(hist)}


def get_glcm_ms_features(gray_patch_int, prefix):
    """
    Rotation-averaged GLCM properties at several pixel distances, on GLCM_MS_LEVELS grey levels.
    """
    q = (gray_patch_int.astype(np.int32) * GLCM_MS_LEVELS // 256).astype(np.uint8)
    glcm = graycomatrix(q, distances=list(GLCM_MS_DISTANCES), angles=list(GLCM_MS_ANGLES),
                        levels=GLCM_MS_LEVELS, symmetric=True, normed=True)
    feats = {}
    for p in GLCM_MS_PROPS:
        if p == 'entropy':
            vals = -np.sum(glcm * np.log2(glcm + 1e-12), axis=(0, 1))
        else:
            vals = graycoprops(glcm, p)
        vals = np.nan_to_num(vals, nan=1.0 if p == 'correlation' else 0.0).mean(axis=1)   # mean over angles
        for d, v in zip(GLCM_MS_DISTANCES, vals):
            feats[f'glcm_ms_{prefix}_d{d}_{p}'] = v
    return feats


def get_lbp_ms_histograms(gray_patch_int, prefix):
    """Uniform LBP histograms (8 neighbours) at several radii."""
    feats = {}
    for r in LBP_MS_RADII:
        lbp = local_binary_pattern(gray_patch_int, LBP_POINTS, r, LBP_METHOD)
        hist, _ = np.histogram(lbp.ravel(), density=True, bins=np.arange(0, LBP_N_BINS + 1))
        feats.update({f'lbp_ms_{prefix}_r{r}_bin_{i}': val for i, val in enumerate(hist)})
    return feats


def get_gradient_features(gray_patch_int, prefix):
    """Mean and standard deviation of the intensity gradient magnitude (edge strength)."""
    gy, gx = np.gradient(gray_patch_int.astype(float))
    mag = np.hypot(gx, gy)
    return {f'grad_{prefix}_mean': mag.mean(), f'grad_{prefix}_std': mag.std()}


def _needed(names):
    """Works out which feature groups must be computed to produce `names`."""
    names = set(names)
    return {
        'spectral': bool(names & set(SPECTRAL_FEATURES)),
        'glcm': {ch for ch in TEXTURE_CHANNELS if any(n.startswith(f'glcm_{ch}_') for n in names)},
        'lbp': {ch for ch in TEXTURE_CHANNELS if any(n.startswith(f'lbp_{ch}_') for n in names)},
        'glcm_ms': {ch for ch in MS_CHANNELS if any(n.startswith(f'glcm_ms_{ch}_') for n in names)},
        'lbp_ms': {ch for ch in LBP_MS_CHANNELS if any(n.startswith(f'lbp_ms_{ch}_') for n in names)},
        'gradient': {ch for ch in MS_CHANNELS if any(n.startswith(f'grad_{ch}_') for n in names)},
    }


def compute_block_features(rgb, lab, channels, needed):
    """Features for one block, given its RGB, CIELAB and 8-bit texture channels."""
    feats = {}
    if needed['spectral']:
        mean_rgb = rgb.reshape(-1, 3).mean(axis=0)
        mean_lab = lab.reshape(-1, 3).mean(axis=0)
        v, l_star = channels['hsv_v'], lab[:, :, 0]
        feats.update({
            'mean_r': mean_rgb[0], 'mean_g': mean_rgb[1], 'mean_b': mean_rgb[2],
            'mean_hsv_v': v.mean(), 'mean_lab_l': l_star.mean(),
            'norm_a_star': mean_lab[1] / (mean_lab[0] + 1e-6),
            'norm_b_star': mean_lab[2] / (mean_lab[0] + 1e-6),
            'std_hsv_v': v.std(), 'var_hsv_v': v.var(),
            'std_lab_l': l_star.std(), 'var_lab_l': l_star.var(),
        })
        feats.update(get_color_invariants(rgb))
    for ch in needed['glcm']:
        feats.update(get_glcm_features(channels[ch], ch))
    for ch in needed['lbp']:
        feats.update(get_lbp_histogram(channels[ch], ch))
    for ch in {ch for ch, _ in needed['glcm_ms']}:
        feats.update(get_glcm_ms_features(channels[ch], ch))
    for ch in {ch for ch, _ in needed['lbp_ms']}:
        feats.update(get_lbp_ms_histograms(channels[ch], ch))
    for ch in needed['gradient']:
        feats.update(get_gradient_features(channels[ch], ch))
    return feats
