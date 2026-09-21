# Generic loader turning a Hugging Face image dataset into a stack of
# postage stamps ready for `galmorph.stats`.
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
    Loads images from a Hugging Face dataset and turns them into a stack of
    square, single-channel postage stamps.

    Parameters
    ----------
    dataset_name: str
        Dataset identifier passed to `datasets.load_dataset` (e.g.
        "org/name" or a local path understood by the `datasets` library).
    split: str
        Dataset split to load.
    image_field: str
        Name of the column holding the images (PIL images, arrays, or
        already-decoded tensors).
    n_samples: int, optional
        If given, only load the first `n_samples` examples (use
        `streaming=True` for very large datasets).
    stamp_size: int
        Output stamp size in pixels. Images are center-cropped or
        zero-padded to this size.
    to_grayscale: {"mean", "first_channel", None}
        How to collapse multi-channel images to a single channel. `None`
        leaves images untouched, which requires them to already be 2D.
    normalize: callable, optional
        Optional function applied to each image array right after loading
        (e.g. to rescale pixel values to flux units). Receives and returns
        a 2D numpy array.
    hf_config: str, optional
        Optional dataset configuration name.
    streaming: bool
        Passed through to `datasets.load_dataset`; use for datasets too
        large to fit in memory.
    seed: int
        Shuffle seed used when subsampling a streaming dataset.
    extra_fields: list[str], optional
        Extra scalar catalog columns to read alongside the images (e.g. a
        magnitude field to use for binned plots). When given, this
        function returns `(stamps, extra)` where `extra` is a
        dict[str, numpy.ndarray] aligned with `stamps`.
    hf_token: str, optional
        Auth token used to access private/gated Hugging Face datasets.
        Defaults to the `HF_TOKEN` environment variable.
    psf_field: str, optional
        Name of a column holding a per-object PSF stamp. When given, it is
        kept at its native size (not cropped/padded to `stamp_size` — PSF
        stamps are typically much smaller than the science stamp, and the
        convolution step is expected to handle the size mismatch itself)
        and returned under the `"psf"` key of the `extra` dict (forcing
        the `(stamps, extra)` return form even if `extra_fields` is not
        given) — used to reconvolve autoencoder reconstructions before
        computing statistics on them.
    noise_map_field: str, optional
        Name of a column holding a per-pixel noise map (e.g. the pixel
        noise standard deviation, aligned with `image_field`). Fit to
        `stamp_size` the same way as the science image and returned under
        the `"noise_map"` key of the `extra` dict (forcing the `(stamps,
        extra)` return form even if `extra_fields` is not given) — used by
        `add_noise` to give otherwise noiseless images (e.g. autoencoder
        reconstructions) a realistic per-pixel noise realization.
    mask_field: str, optional
        Name of a column holding a per-pixel validity mask aligned with
        `image_field`, using the dataset's own convention: 1 = valid
        pixel, 0 = defective/corrupted pixel (e.g. a cosmic ray hit or
        other detector defect). Fit to `stamp_size` the same way as the
        science image, with padding added outside the original stamp
        treated as defective (0). Before being returned, the mask is
        flipped to the opposite convention -- nonzero = bad pixel -- since
        that is what `galmorph.stats.moments`/`morph_stats` expect and
        what GalSim's own `badpix` argument requires; this flip is handled
        here so callers never have to think about it. Returned under the
        `"mask"` key of the `extra` dict (forcing the `(stamps, extra)`
        return form even if `extra_fields` is not given) -- pass it as the
        corresponding entry of `galmorph.pipeline.compute_statistics`'s
        `masks` argument so defective pixels are excluded/estimated
        instead of trusted as real zero flux.
    test_size: float, optional
        If given, `split` is re-split with `Dataset.train_test_split(
        test_size=test_size, seed=split_seed)` and only its `"test"` part is
        kept, before `n_samples` is applied. With `test_size=0.1` and the
        default `split_seed=42` this is exactly the held-out set Train-AE
        trains its models against. The split is shuffled, so a
        `"train[90%:]"` slice does not select the same objects. Cannot be
        combined with `streaming=True`.
    split_seed: int
        Seed of the `test_size` split.

    Returns
    -------
    numpy.ndarray, shape (N, stamp_size, stamp_size)
        Or `(stamps, extra)` if `extra_fields`, `psf_field`, `noise_map_field`
        and/or `mask_field` is given.
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
            valid = _fit_to_stamp(valid, stamp_size, fill=0.0)  # padding = defective, dataset's own convention
            extra["mask"].append((valid == 0).astype(np.float64))  # flip to nonzero = bad, for stats.py/GalSim
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
    Adds a white-noise realization to `images`, scaled by a per-pixel
    `noise_map` (e.g. the `"noise_map"` stamps returned by
    `load_hf_stamps`).

    This is meant for otherwise noiseless images, such as autoencoder
    reconstructions: the CAS/Gini-M20/MID indicators (`galmorph.stats.
    morph_stats`) estimate their segmentation threshold and S/N from the
    background pixel scatter, which is degenerate on a noise-free image, so
    reconstructions need a realistic noise realization before those
    statistics are meaningful to compare against the real images.

    Parameters
    ----------
    images: array_like, shape (N, H, W)
    noise_map: array_like, shape (N, H, W)
        Per-pixel noise standard deviation, aligned with `images`.
    seed: int, optional
        Seed for the white noise draw, for reproducibility.

    Returns
    -------
    numpy.ndarray, shape (N, H, W)
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
    if hasattr(img, "convert") and hasattr(img, "size"):  # PIL.Image
        return np.asarray(img, dtype=np.float64)
    return np.asarray(img, dtype=np.float64)


def _collapse_channels(img, mode):
    if img.ndim == 2 or mode is None:
        return img
    if mode == "mean":
        return img.mean(axis=-1)
    if mode == "first_channel":
        return img[..., 0]
    raise ValueError("Unknown to_grayscale mode: %r" % mode)


def _fit_to_stamp(img, stamp_size, fill=0.0):
    """Center-crops or pads a 2D image to (stamp_size, stamp_size), filling
    any padding with `fill` (0 for science/noise images, 1 -- "bad/no data"
    -- for masks)."""
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
