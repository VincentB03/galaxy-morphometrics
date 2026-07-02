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

R_INDICATORS_DIR = os.path.join(os.path.dirname(__file__), "r_indicators")


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


def moments(images, scale=0.03, stamp_size=None):
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

    Returns
    -------
    astropy.table.Table with columns: amp, sigma_e, e, e1, e2, g, g1, g2,
    rho4, flag
    """
    images = _as_stack(images)
    if stamp_size is None:
        stamp_size = images.shape[-1]

    sigma, e, e1, e2, g, g1, g2, flag, amp, rho4 = ([] for _ in range(10))

    for i in range(len(images)):
        image = galsim.Image(np.ascontiguousarray(images[i], dtype=np.float64), scale=scale)
        shape = image.FindAdaptiveMom(
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


def morph_stats(images):
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

    flag = []
    rows = []
    for i in range(len(images)):
        im = np.ascontiguousarray(images[i], dtype=np.float64)
        ret = compute_statistics_single(im)
        flag.append(bool(ret[0][0]))
        with localconverter(ro.default_converter + pandas2ri.converter):
            rows.append(ro.conversion.rpy2py(ret[1]))

    tab = Table.from_pandas(pd.concat(rows))
    tab["flag"] = flag
    return tab
