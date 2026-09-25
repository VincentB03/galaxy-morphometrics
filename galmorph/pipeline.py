# HSM moments + R morphological indicators for one or several named sets of
# postage stamps (e.g. "real" vs "reconstruction").
from multiprocessing import Pool

import numpy as np
from astropy.table import Table, join, vstack

from .stats import moments, morph_stats


def _chunk(images, n_chunks):
    return np.array_split(np.asarray(images), n_chunks)


def _chunk_or_none(array, n_chunks):
    return [None] * n_chunks if array is None else _chunk(array, n_chunks)


def _moments_chunk(args):
    images, masks, pixel_scale = args
    return moments(images, scale=pixel_scale, masks=masks)


def _morph_stats_chunk(args):
    images, masks = args
    return morph_stats(images, masks=masks)


def compute_statistics_single(
    images,
    pixel_scale=0.03,
    morph_crop=None,
    pool_size=None,
    compute_morph=True,
    masks=None,
):
    """
    Computes HSM moments and, optionally, CAS/Gini-M20/MID for one stack of
    postage stamps.

    Parameters
    ----------
    images: array_like, shape (N, H, W)
    pixel_scale: float
        Pixel scale in arcsec/pixel.
    morph_crop: int, optional
        Center-crops stamps (and masks) to this size before the R indicators.
    pool_size: int, optional
        Number of worker processes. None disables multiprocessing.
    compute_morph: bool
        Computes the R indicators. False if R/SDMTools is not installed.
    masks: array_like, shape (N, H, W), optional
        Bad-pixel masks (nonzero = bad), see `stats.moments` and
        `stats.morph_stats`.

    Returns
    -------
    astropy.table.Table with one row per image. `flag` is True only when every
    requested statistic succeeded.
    """
    images = np.asarray(images)
    ident = np.arange(len(images))
    if masks is not None:
        masks = np.asarray(masks)
        if masks.shape != images.shape:
            raise ValueError("masks and images must have the same shape (%r vs %r)" % (masks.shape, images.shape))

    n_chunks = pool_size or 1
    chunks = _chunk(images, n_chunks)
    mask_chunks = _chunk_or_none(masks, n_chunks)
    moments_args = list(zip(chunks, mask_chunks, [pixel_scale] * n_chunks))

    if pool_size:
        with Pool(pool_size) as p:
            hsm_table = vstack(p.map(_moments_chunk, moments_args))
    else:
        hsm_table = vstack([_moments_chunk(a) for a in moments_args])
    hsm_table["IDENT"] = ident

    if not compute_morph:
        hsm_table.rename_column("flag", "flag_moments")
        hsm_table["flag"] = hsm_table["flag_moments"]
        return hsm_table

    morph_images = images
    morph_masks = masks
    if morph_crop is not None:
        h, w = images.shape[1], images.shape[2]
        y0 = (h - morph_crop) // 2
        x0 = (w - morph_crop) // 2
        morph_images = images[:, y0 : y0 + morph_crop, x0 : x0 + morph_crop]
        if morph_masks is not None:
            morph_masks = morph_masks[:, y0 : y0 + morph_crop, x0 : x0 + morph_crop]
    morph_chunks = _chunk(morph_images, n_chunks)
    morph_mask_chunks = _chunk_or_none(morph_masks, n_chunks)
    morph_args = list(zip(morph_chunks, morph_mask_chunks))

    if pool_size:
        with Pool(pool_size) as p:
            stats_table = vstack(p.map(_morph_stats_chunk, morph_args))
    else:
        stats_table = vstack([_morph_stats_chunk(a) for a in morph_args])
    stats_table["IDENT"] = ident

    table = join(hsm_table, stats_table, keys=["IDENT"], table_names=["moments", "morph"])
    table["flag"] = table["flag_moments"] & table["flag_morph"]
    return table


def compute_statistics(datasets, pixel_scale=0.03, morph_crop=None, pool_size=None, compute_morph=True, masks=None):
    """
    Runs `compute_statistics_single` on each entry of `datasets` (name ->
    stack of stamps). `masks` maps a dataset name to its bad-pixel masks;
    datasets missing from it are computed without masks.

    Returns
    -------
    dict[str, astropy.table.Table]
    """
    masks = masks or {}
    return {
        name: compute_statistics_single(
            images,
            pixel_scale=pixel_scale,
            morph_crop=morph_crop,
            pool_size=pool_size,
            compute_morph=compute_morph,
            masks=masks.get(name),
        )
        for name, images in datasets.items()
    }
