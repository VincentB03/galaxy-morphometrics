# Ties together moments + morphological statistics computation for one or
# several named sets of postage stamps (e.g. "real" vs "reconstruction").
from multiprocessing import Pool

import numpy as np
from astropy.table import Table, join, vstack

from .stats import moments, morph_stats


def _chunk(images, n_chunks):
    return np.array_split(np.asarray(images), n_chunks)


def compute_statistics_single(
    images,
    pixel_scale=0.03,
    morph_crop=None,
    pool_size=None,
    compute_morph=True,
):
    """
    Computes HSM moments and (optionally) CAS/Gini-M20/MID statistics for a
    single stack of postage stamps.

    Parameters
    ----------
    images: array_like, shape (N, H, W)
    pixel_scale: float
        Pixel scale in arcsec/pixel, used by the HSM moments.
    morph_crop: int, optional
        If given, crop each stamp to a `morph_crop` x `morph_crop` window
        around the center before computing the R-based morphological
        indicators (the original paper crops 128x128 stamps to 64x64).
    pool_size: int, optional
        Number of worker processes. None disables multiprocessing.
    compute_morph: bool
        Whether to compute the R-based CAS/Gini-M20/MID indicators. Set to
        False if R/rpy2/SDMTools are not installed.

    Returns
    -------
    astropy.table.Table with one row per image, and a `flag` column that
    is True only when every requested statistic succeeded.
    """
    images = np.asarray(images)
    ident = np.arange(len(images))

    n_chunks = pool_size or 1
    chunks = _chunk(images, n_chunks)

    if pool_size:
        with Pool(pool_size) as p:
            hsm_table = vstack(p.map(moments, chunks))
    else:
        hsm_table = vstack([moments(c) for c in chunks])
    hsm_table["IDENT"] = ident

    if not compute_morph:
        hsm_table.rename_column("flag", "flag_moments")
        hsm_table["flag"] = hsm_table["flag_moments"]
        return hsm_table

    morph_images = images
    if morph_crop is not None:
        h, w = images.shape[1], images.shape[2]
        y0 = (h - morph_crop) // 2
        x0 = (w - morph_crop) // 2
        morph_images = images[:, y0 : y0 + morph_crop, x0 : x0 + morph_crop]
    morph_chunks = _chunk(morph_images, n_chunks)

    if pool_size:
        with Pool(pool_size) as p:
            stats_table = vstack(p.map(morph_stats, morph_chunks))
    else:
        stats_table = vstack([morph_stats(c) for c in morph_chunks])
    stats_table["IDENT"] = ident

    table = join(hsm_table, stats_table, keys=["IDENT"], table_names=["moments", "morph"])
    table["flag"] = table["flag_moments"] & table["flag_morph"]
    return table


def compute_statistics(datasets, pixel_scale=0.03, morph_crop=None, pool_size=None, compute_morph=True):
    """
    Computes statistics for several named datasets at once.

    Parameters
    ----------
    datasets: dict[str, array_like]
        Mapping from dataset name (e.g. "real", "reconstruction") to a
        stack of postage stamps of identical shape.

    Returns
    -------
    dict[str, astropy.table.Table]
    """
    return {
        name: compute_statistics_single(
            images,
            pixel_scale=pixel_scale,
            morph_crop=morph_crop,
            pool_size=pool_size,
            compute_morph=compute_morph,
        )
        for name, images in datasets.items()
    }
