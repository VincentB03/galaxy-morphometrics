# Hugging Face image dataset -> stack of postage stamps for `galmorph.stats`.
import os

import numpy as np


def load_hf_stamps(
    dataset_name,
    split="train",
    image_field="image",
    n_samples=None,
    stamp_size=128,
    to_grayscale="mean",
    normalize=None,
    hf_config=None,
    streaming=False,
    seed=0,
    extra_fields=None,
    hf_token=None,
    psf_field=None,
    noise_map_field=None,
    mask_field=None,
    test_size=None,
    split_seed=42,
):
    """
    Loads a Hugging Face image dataset as a stack of square, single-channel
    postage stamps. Images, noise maps and masks are center-cropped or
    zero-padded to `stamp_size`; PSF stamps keep their native size.

    Parameters
    ----------
    dataset_name, hf_config, split, streaming:
        Passed to `datasets.load_dataset`.
    image_field: str
        Image column.
    n_samples: int, optional
        Keeps the first `n_samples` rows. With `streaming`, takes them from a
        shuffle buffer seeded by `seed` instead.
    stamp_size: int
        Output stamp size in pixels.
    to_grayscale: {"mean", "first_channel", None}
        How to collapse multi-channel images. None requires 2D images.
    normalize: callable, optional
        Applied to each 2D image right after loading.
    extra_fields: list[str], optional
        Scalar catalog columns to return alongside the stamps.
    hf_token: str, optional
        Token for private/gated datasets (default: `HF_TOKEN` env variable).
    psf_field, noise_map_field: str, optional
        Per-object PSF stamp and per-pixel noise std columns.
    mask_field: str, optional
        Per-pixel validity mask column (1 = valid, 0 = bad). Returned flipped
        to nonzero = bad, the convention of `galmorph.stats` and GalSim.
        Padding counts as bad.
    test_size: float, optional
        Keeps only the "test" part of `train_test_split(test_size,
        seed=split_seed)`, before `n_samples` is applied. 0.1 with seed 42 is
        Train-AE's held-out set. Not compatible with `streaming`.
    split_seed: int
        Seed of the `test_size` split.

    Returns
    -------
    stamps: numpy.ndarray, shape (N, stamp_size, stamp_size)
        Or `(stamps, extra)` if any of `extra_fields`, `psf_field`,
        `noise_map_field` or `mask_field` is given. `extra` maps each extra
        field, and "psf", "noise_map", "mask", to an array aligned with
        `stamps`.
    """
    from datasets import load_dataset

    token = hf_token if hf_token is not None else os.environ.get("HF_TOKEN")
    ds = load_dataset(dataset_name, hf_config, split=split, streaming=streaming, token=token)

    if test_size is not None:
        if streaming:
            raise ValueError("test_size needs the whole split to shuffle it and cannot be combined with streaming")
        ds = ds.train_test_split(test_size=test_size, seed=split_seed)["test"]

    if n_samples is not None:
        if streaming:
            ds = ds.shuffle(seed=seed, buffer_size=max(10 * n_samples, 1000))
            ds = ds.take(n_samples)
        else:
            n_samples = min(n_samples, len(ds))
            ds = ds.select(range(n_samples))

    stamps = []
    extra = {f: [] for f in (extra_fields or [])}
    if psf_field:
        extra["psf"] = []
    if noise_map_field:
        extra["noise_map"] = []
    if mask_field:
        extra["mask"] = []
    for example in ds:
        img = _to_array(example[image_field])
        img = _collapse_channels(img, to_grayscale)
        if normalize is not None:
            img = normalize(img)
        img = _fit_to_stamp(img, stamp_size)
        stamps.append(img)
        if psf_field:
            psf = _collapse_channels(_to_array(example[psf_field]), to_grayscale)
            extra["psf"].append(psf)
        if noise_map_field:
            noise_map = _collapse_channels(_to_array(example[noise_map_field]), to_grayscale)
            extra["noise_map"].append(_fit_to_stamp(noise_map, stamp_size))
        if mask_field:
            valid = _collapse_channels(_to_array(example[mask_field]), to_grayscale)
            valid = _fit_to_stamp(valid, stamp_size, fill=0.0)  # padding counts as bad
            extra["mask"].append((valid == 0).astype(np.float64))  # flip to nonzero = bad
        for f in (extra_fields or []):
            extra[f].append(example[f])

    stamps = np.stack(stamps).astype(np.float64)
    if extra_fields or psf_field or noise_map_field or mask_field:
        return stamps, {
            k: (np.stack(v) if k in ("psf", "noise_map", "mask") else np.asarray(v)) for k, v in extra.items()
        }
    return stamps


def add_noise(images, noise_map, seed=None):
    """
    Adds white Gaussian noise of per-pixel standard deviation `noise_map`
    (shape (N, H, W), aligned with `images`) to `images`.

    Meant for noise-free images such as autoencoder reconstructions: the R
    indicators estimate their segmentation threshold and S/N from the
    background scatter, which is degenerate without noise.
    """
    images = np.asarray(images)
    noise_map = np.asarray(noise_map)
    if noise_map.shape != images.shape:
        raise ValueError(
            "noise_map and images must have the same shape (%r vs %r)" % (noise_map.shape, images.shape)
        )
    rng = np.random.default_rng(seed)
    return images + rng.standard_normal(images.shape) * noise_map


def _to_array(img):
    return np.asarray(img, dtype=np.float64)  # PIL images, arrays, tensors


def _collapse_channels(img, mode):
    if img.ndim == 2 or mode is None:
        return img
    if mode == "mean":
        return img.mean(axis=-1)
    if mode == "first_channel":
        return img[..., 0]
    raise ValueError("Unknown to_grayscale mode: %r" % mode)


def _fit_to_stamp(img, stamp_size, fill=0.0):
    """Center-crops or pads a 2D image to (stamp_size, stamp_size), padding with `fill`."""
    h, w = img.shape
    out = np.full((stamp_size, stamp_size), fill, dtype=img.dtype)

    src_y0 = max(0, (h - stamp_size) // 2)
    src_x0 = max(0, (w - stamp_size) // 2)
    src_y1 = src_y0 + min(h, stamp_size)
    src_x1 = src_x0 + min(w, stamp_size)
    crop = img[src_y0:src_y1, src_x0:src_x1]

    dst_y0 = max(0, (stamp_size - h) // 2)
    dst_x0 = max(0, (stamp_size - w) // 2)
    out[dst_y0 : dst_y0 + crop.shape[0], dst_x0 : dst_x0 + crop.shape[1]] = crop
    return out
