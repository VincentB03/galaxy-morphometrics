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
        fitted to `stamp_size` the same way as `image_field` and returned
        under the `"psf"` key of the `extra` dict (forcing the `(stamps,
        extra)` return form even if `extra_fields` is not given) — used to
        reconvolve autoencoder reconstructions before computing statistics
        on them.

    Returns
    -------
    numpy.ndarray, shape (N, stamp_size, stamp_size)
        Or `(stamps, extra)` if `extra_fields` and/or `psf_field` is given.
    """
    from datasets import load_dataset

    token = hf_token if hf_token is not None else os.environ.get("HF_TOKEN")
    ds = load_dataset(dataset_name, hf_config, split=split, streaming=streaming, token=token)

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
    for example in ds:
        img = _to_array(example[image_field])
        img = _collapse_channels(img, to_grayscale)
        if normalize is not None:
            img = normalize(img)
        img = _fit_to_stamp(img, stamp_size)
        stamps.append(img)
        if psf_field:
            psf = _fit_to_stamp(_collapse_channels(_to_array(example[psf_field]), to_grayscale), stamp_size)
            extra["psf"].append(psf)
        for f in (extra_fields or []):
            extra[f].append(example[f])

    stamps = np.stack(stamps).astype(np.float64)
    if extra_fields or psf_field:
        return stamps, {k: (np.stack(v) if k == "psf" else np.asarray(v)) for k, v in extra.items()}
    return stamps


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


def _fit_to_stamp(img, stamp_size):
    """Center-crops or zero-pads a 2D image to (stamp_size, stamp_size)."""
    h, w = img.shape
    out = np.zeros((stamp_size, stamp_size), dtype=img.dtype)

    src_y0 = max(0, (h - stamp_size) // 2)
    src_x0 = max(0, (w - stamp_size) // 2)
    src_y1 = src_y0 + min(h, stamp_size)
    src_x1 = src_x0 + min(w, stamp_size)
    crop = img[src_y0:src_y1, src_x0:src_x1]

    dst_y0 = max(0, (stamp_size - h) // 2)
    dst_x0 = max(0, (stamp_size - w) // 2)
    out[dst_y0 : dst_y0 + crop.shape[0], dst_x0 : dst_x0 + crop.shape[1]] = crop
    return out
