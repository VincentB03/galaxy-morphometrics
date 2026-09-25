# Morphometric statistics on galaxy postage stamps:
#   moments()      HSM adaptive moments (GalSim)
#   morph_stats()  CAS / Gini-M20 / MID, via the R code in r_indicators/
import os
import numpy as np
import pandas as pd
import galsim
from astropy.table import Table
from scipy.ndimage import median_filter

R_INDICATORS_DIR = os.path.join(os.path.dirname(__file__), "r_indicators")

# Columns returned by the R code, used for the placeholder row of skipped stamps
_MORPH_COLUMNS = [
    "M_level", "M", "M_level_o", "M_o", "M_level_p", "M_p",
    "I", "D", "axmax", "axmin", "angle", "sn", "size", "Gini", "M20", "C", "A",
]


def _as_stack(images):
    """Returns a single (H, W) image or an (N, H, W) stack as an (N, H, W) array."""
    images = np.asarray(images)
    if images.ndim == 2:
        images = images[np.newaxis]
    elif images.ndim != 3:
        raise ValueError("Expected a (H, W) or (N, H, W) array, got shape %r" % (images.shape,))
    return images


def _as_mask_stack(masks, images):
    """Returns the optional bad-pixel masks (nonzero = bad) with the shape of `images`."""
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
    scale: float
        Pixel scale in arcsec/pixel.
    stamp_size: int, optional
        Centroid guess is (stamp_size // 2, stamp_size // 2). Defaults to the
        image size.
    masks: array_like, shape (N, H, W), optional
        Bad-pixel masks (nonzero = bad), excluded from the fit through
        GalSim's `badpix` instead of counting as zero flux.

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
    """Replaces masked pixels with their local median. The R code has no notion
    of missing data, and a zero-flux hole biases the indicators more."""
    filled = median_filter(image, size=size)
    return np.where(bad, filled, image)


def morph_stats(images, masks=None, max_masked_frac=0.10):
    """
    Computes the CAS (Concentration, Asymmetry), Gini/M20 and MID (Multimode,
    Intensity, Deviation) indicators with the R code, via rpy2. Requires R
    with SDMTools (see README).

    Parameters
    ----------
    images: array_like, shape (N, H, W)
        Postage stamps, ideally cropped around the galaxy.
    masks: array_like, shape (N, H, W), optional
        Bad-pixel masks (nonzero = bad). Bad pixels are filled with
        `_fill_masked` before the measurement.
    max_masked_frac: float
        Stamps with a larger masked fraction are skipped (flag False, -9
        values).

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
