# Morphometric statistics on postage-stamp galaxy images.
#
# `moments()` computes HSM (adaptive moments) shape statistics with GalSim.
# `morph_stats()` computes the CAS / Gini-M20 / MID indicators via the R
# routines in `r_indicators/`, ported from deepgal/validation.
import os
import numpy as np
import pandas as pd
import galsim
from astropy.table import Table
from scipy.ndimage import median_filter

R_INDICATORS_DIR = os.path.join(os.path.dirname(__file__), "r_indicators")

# Columns returned by the R `compute_statistics_single` routine, used to
# build a placeholder row for stamps skipped in `morph_stats` because too
# much of them is masked (see `max_masked_frac`).
_MORPH_COLUMNS = [
    "M_level", "M", "M_level_o", "M_o", "M_level_p", "M_p",
    "I", "D", "axmax", "axmin", "angle", "sn", "size", "Gini", "M20", "C", "A",
]


def _as_stack(images):
    """Normalizes input to an (N, H, W) array of images. Accepts a single
    (H, W) image or an already-batched (N, H, W) stack, which is how every
    caller in this package (`pipeline.py`, `run_morphometrics.py`) passes
    images."""
    images = np.asarray(images)
    if images.ndim == 2:
        images = images[np.newaxis]
    elif images.ndim != 3:
        raise ValueError("Expected a (H, W) or (N, H, W) array, got shape %r" % (images.shape,))
    return images


def _as_mask_stack(masks, images):
    """Normalizes an optional bad-pixel mask to match `images`'s (N, H, W)
    shape. Any nonzero value marks a bad pixel (e.g. a cosmic ray hit or
    other detector defect)."""
    if masks is None:
        return None
    masks = _as_stack(np.asarray(masks))
    if masks.shape != images.shape:
        raise ValueError("masks and images must have the same shape (%r vs %r)" % (masks.shape, images.shape))
    return masks


def moments(images, scale=0.03, stamp_size=None, masks=None):
    """
    Computes HSM adaptive moments for a stack of images.

    Parameters
    ----------
    images: array_like, shape (N, H, W)
        Postage stamps.
    scale: float
        Pixel scale in arcsec/pixel.
    stamp_size: int, optional
        Size of the postage stamp, used as the centroid guess. Defaults to
        the image size.
    masks: array_like, shape (N, H, W), optional
        Bad-pixel masks aligned with `images` (nonzero = bad pixel, e.g. a
        cosmic ray hit). When given, bad pixels are excluded from the HSM
        fit via GalSim's `badpix` argument, instead of being treated as
        real zero flux -- which would otherwise bias the measured size and
        ellipticity.

    Returns
    -------
    astropy.table.Table with columns: amp, sigma_e, e, e1, e2, g, g1, g2,
    rho4, flag
    """
    images = _as_stack(images)
    if stamp_size is None:
        stamp_size = images.shape[-1]
    masks = _as_mask_stack(masks, images)

    sigma, e, e1, e2, g, g1, g2, flag, amp, rho4 = ([] for _ in range(10))

    for i in range(len(images)):
        image = galsim.Image(np.ascontiguousarray(images[i], dtype=np.float64), scale=scale)
        badpix = None
        if masks is not None:
            badpix = galsim.Image(np.ascontiguousarray(masks[i] != 0, dtype=np.int16), scale=scale)
        shape = image.FindAdaptiveMom(
            badpix=badpix,
            guess_centroid=galsim.PositionD(stamp_size // 2, stamp_size // 2),
            strict=False,
        )
        amp.append(shape.moments_amp)
        sigma.append(shape.moments_sigma)
        rho4.append(shape.moments_rho4)
        e.append(shape.observed_shape.e)
        e1.append(shape.observed_shape.e1)
        e2.append(shape.observed_shape.e2)
        g.append(shape.observed_shape.g)
        g1.append(shape.observed_shape.g1)
        g2.append(shape.observed_shape.g2)
        flag.append(shape.error_message == "")

    return Table(
        {
            "amp": amp,
            "sigma_e": sigma,
            "e": e,
            "e1": e1,
            "e2": e2,
            "g": g,
            "g1": g1,
            "g2": g2,
            "rho4": rho4,
            "flag": flag,
        }
    )


def _fill_masked(image, bad, size=5):
    """Fills masked pixels with the median of their local neighborhood.

    This is only a plausible local estimate, not the true pixel value -- it
    exists because the R statistics below have no notion of missing data
    and would otherwise see a sharp zero-flux hole, which biases
    Gini/M20/Asymmetry/Multimode more than a smooth local estimate does.
    """
    filled = median_filter(image, size=size)
    return np.where(bad, filled, image)


def morph_stats(images, masks=None, max_masked_frac=0.10):
    """
    Computes CAS (Concentration, Asymmetry, Smoothness), Gini/M20 and MID
    (Multimode, Intensity, Deviation) morphological indicators using the R
    implementation from Freeman et al. / Lotz et al., via rpy2.

    Requires R with the `SDMTools` package installed (see README).

    Parameters
    ----------
    images: array_like, shape (N, H, W)
        Postage stamps. For best results, crop to the galaxy-centered
        region (the original paper uses 64x64 stamps cropped from 128x128).
    masks: array_like, shape (N, H, W), optional
        Bad-pixel masks aligned with `images` (nonzero = bad pixel). The R
        routines have no notion of missing data, so masked pixels are
        filled in with a local median estimate before the statistics are
        computed (see `_fill_masked`), rather than left at the raw zero
        value in `images` -- which would otherwise read as spurious
        structure to the Asymmetry/Multimode/Gini/M20 indicators. Stamps
        with a masked fraction above `max_masked_frac` are skipped entirely
        (flagged False) instead of being measured on mostly fabricated
        data.
    max_masked_frac: float
        Maximum fraction of masked pixels tolerated per stamp before it is
        skipped rather than filled in. Ignored if `masks` is None.

    Returns
    -------
    astropy.table.Table with columns: M_level, M, M_level_o, M_o,
    M_level_p, M_p, I, D, axmax, axmin, angle, sn, size, Gini, M20, C, A,
    flag
    """
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri, pandas2ri
    from rpy2.robjects.conversion import localconverter

    ro.r('source("%s/interface.R", chdir=T)' % R_INDICATORS_DIR)
    compute_statistics_single = ro.r.compute_statistics_single
    pandas2ri.activate()
    numpy2ri.activate()

    images = _as_stack(images)
    masks = _as_mask_stack(masks, images)

    flag = []
    rows = []
    for i in range(len(images)):
        im = np.ascontiguousarray(images[i], dtype=np.float64)

        if masks is not None:
            bad = masks[i] != 0
            if bad.mean() > max_masked_frac:
                flag.append(False)
                rows.append(pd.DataFrame([dict.fromkeys(_MORPH_COLUMNS, -9.0)]))
                continue
            if bad.any():
                im = _fill_masked(im, bad)

        ret = compute_statistics_single(im)
        flag.append(bool(ret[0][0]))
        with localconverter(ro.default_converter + pandas2ri.converter):
            rows.append(ro.conversion.rpy2py(ret[1]))

    tab = Table.from_pandas(pd.concat(rows))
    tab["flag"] = flag
    return tab
