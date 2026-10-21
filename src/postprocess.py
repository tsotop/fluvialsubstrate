import numpy as np
from scipy.ndimage import uniform_filter

"""
Post-processing of block-level class probabilities.

Shared by inference (whole map) and the benchmark (windows around validation points),
so both produce the same labels for the same blocks.
"""

NODATA = 255


def neighbourhood_mean(P, size):
    """
    Mean class probabilities over a size x size block window, using only blocks that have a
    prediction (P is NaN elsewhere). Returns NaN where the window holds no prediction.
    """
    have = ~np.isnan(P[0])
    num = uniform_filter(np.where(have, P, 0.0), size=(1, size, size), mode='constant')
    den = uniform_filter(have.astype(float), size=size, mode='constant')
    with np.errstate(invalid='ignore', divide='ignore'):
        out = num / den
    out[:, den < 1e-9] = np.nan
    return out


def finalize(P, valid_fraction, smoothing=1, fill_min_valid=0.5):
    """
    Turns block probabilities into class labels and a confidence value.

    Parameters
    ----------
    P : (K, H, W) float, class probabilities; NaN for blocks without a prediction.
    valid_fraction : (H, W) fraction of valid image pixels per block (0 outside the ROI).
    smoothing : window size (blocks) for averaging probabilities; 1 disables smoothing.
    fill_min_valid : blocks without a prediction but with at least this fraction of valid
        pixels (typically mosaic-edge blocks) take the mean of their 3 x 3 neighbours.
        None disables filling.

    Returns
    -------
    labels : (H, W) uint8, NODATA where no label could be assigned.
    confidence : (H, W) float, probability of the assigned class (NaN where NODATA).
    """
    have = ~np.isnan(P[0])
    S = neighbourhood_mean(P, smoothing) if smoothing > 1 else P.copy()
    S[:, ~have] = np.nan
    if fill_min_valid is not None:
        fill = ~have & (valid_fraction >= fill_min_valid)
        if fill.any():
            S[:, fill] = neighbourhood_mean(P, 3)[:, fill]

    assigned = ~np.isnan(S[0])
    labels = np.full(S.shape[1:], NODATA, dtype='uint8')
    confidence = np.full(S.shape[1:], np.nan)
    if assigned.any():
        Sa = S[:, assigned]
        labels[assigned] = Sa.argmax(axis=0)
        confidence[assigned] = Sa.max(axis=0)
    return labels, confidence
