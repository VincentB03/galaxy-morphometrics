#!/usr/bin/env python
"""
Computes galaxy morphometric statistics (HSM moments + CAS/Gini-M20/MID
indicators) and reproduces the comparison plots from deep_galaxy_models's
deepgal/validation, on your own data.

Typical workflow:
  1. Load a stack of postage stamps from a Hugging Face image dataset.
  2. Optionally reconstruct them through a pretrained autoencoder
     (implement `galmorph.autoencoder.Autoencoder` for your model).
  3. Compute statistics for "real" and "reconstruction" (or any other named
     sets of images you build yourself).
  4. Save the catalogs and render every comparison plot.

Example:
    python run_morphometrics.py \\
        --dataset your-org/your-galaxy-dataset --split train \\
        --image-field image --n-samples 2000 --stamp-size 128 \\
        --autoencoder galmorph.autoencoder:TFHubVAEAutoencoder \\
        --encoder-path modules/vae_16/encoder --decoder-path modules/vae_16/decoder \\
        --out-dir results

Without --autoencoder, only the "real" dataset is analyzed (distribution
plots only, no reconstruction-error plots).
"""
import argparse
import importlib
import os

import numpy as np
from astropy.table import Table

from galmorph.data import add_noise, load_hf_stamps
from galmorph.pipeline import compute_statistics
from galmorph.plotting import make_all_plots


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    g_data = p.add_argument_group("data")
    g_data.add_argument("--dataset", required=True, help="Hugging Face dataset name or path")
    g_data.add_argument("--hf-config", default=None, help="Optional dataset configuration name")
    g_data.add_argument("--split", default="train")
    g_data.add_argument("--image-field", default="image")
    g_data.add_argument("--n-samples", type=int, default=2000)
    g_data.add_argument("--stamp-size", type=int, default=128)
    g_data.add_argument("--pixel-scale", type=float, default=0.03, help="arcsec/pixel")
    g_data.add_argument("--streaming", action="store_true")
    g_data.add_argument(
        "--binning-field", default=None,
        help="Optional numeric catalog column (e.g. magnitude) used for the "
             "ellipticity/rho4-vs-binning plots",
    )
    g_data.add_argument(
        "--hf-token", default=os.environ.get("HF_TOKEN"),
        help="Auth token for private/gated Hugging Face datasets. "
             "Defaults to the HF_TOKEN environment variable.",
    )
    g_data.add_argument(
        "--psf-field", default=None,
        help="Optional per-object PSF stamp column, fitted to --stamp-size "
             "like --image-field. Needed by autoencoders (e.g. "
             "WandBGalaxyAutoencoder) that reconvolve their reconstruction "
             "with the PSF before statistics are computed on it.",
    )
    g_data.add_argument(
        "--noise-map-field", default=None,
        help="Optional per-object noise map column (per-pixel noise "
             "standard deviation), fitted to --stamp-size like "
             "--image-field. When given together with --autoencoder, white "
             "noise scaled by this map is added to the reconstructed "
             "images, since they otherwise come out noise-free and the "
             "CAS/Gini-M20/MID indicators need a realistic S/N to be "
             "meaningful.",
    )
    g_data.add_argument(
        "--noise-seed", type=int, default=0,
        help="Seed for the white noise draw used by --noise-map-field.",
    )

    g_ae = p.add_argument_group("autoencoder (optional)")
    g_ae.add_argument(
        "--autoencoder", default=None,
        help="'module.path:ClassName' of an Autoencoder subclass, "
             "e.g. galmorph.autoencoder:TFHubVAEAutoencoder",
    )
    g_ae.add_argument("--encoder-path", default=None, help="Passed as first positional arg to the autoencoder class")
    g_ae.add_argument("--decoder-path", default=None, help="Passed as second positional arg to the autoencoder class")

    g_stats = p.add_argument_group("statistics")
    g_stats.add_argument("--pool-size", type=int, default=None, help="Worker processes for stats computation")
    g_stats.add_argument("--morph-crop", type=int, default=None, help="Crop stamps to this size before R stats (e.g. 64)")
    g_stats.add_argument("--skip-r", action="store_true", help="Skip CAS/Gini-M20/MID (no R/rpy2 required)")

    p.add_argument("--out-dir", default="./results")
    return p.parse_args()


def build_autoencoder(spec, encoder_path, decoder_path):
    module_name, class_name = spec.split(":")
    cls = getattr(importlib.import_module(module_name), class_name)
    args = [a for a in (encoder_path, decoder_path) if a is not None]
    return cls(*args)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    plots_dir = os.path.join(args.out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    extra_fields = [args.binning_field] if args.binning_field else None
    loaded = load_hf_stamps(
        args.dataset,
        split=args.split,
        image_field=args.image_field,
        n_samples=args.n_samples,
        stamp_size=args.stamp_size,
        hf_config=args.hf_config,
        streaming=args.streaming,
        extra_fields=extra_fields,
        hf_token=args.hf_token,
        psf_field=args.psf_field,
        noise_map_field=args.noise_map_field,
    )
    if extra_fields or args.psf_field or args.noise_map_field:
        real_images, extra = loaded
        binning_values = {"real": extra[args.binning_field]} if args.binning_field else None
        psf_images = extra["psf"] if args.psf_field else None
        noise_map = extra["noise_map"] if args.noise_map_field else None
    else:
        real_images = loaded
        binning_values = None
        psf_images = None
        noise_map = None
    print("Loaded %d postage stamps of size %dx%d" % (len(real_images), args.stamp_size, args.stamp_size))

    datasets = {"real": real_images}
    reference_name = None

    if args.autoencoder:
        print("Reconstructing images with %s" % args.autoencoder)
        ae = build_autoencoder(args.autoencoder, args.encoder_path, args.decoder_path)
        recon_kwargs = {"psf": psf_images} if psf_images is not None else {}
        datasets["reconstruction"] = ae.reconstruct(real_images, **recon_kwargs)
        if noise_map is not None:
            print("Adding white noise scaled by --noise-map-field to the reconstruction")
            datasets["reconstruction"] = add_noise(datasets["reconstruction"], noise_map, seed=args.noise_seed)
        reference_name = "real"
        if binning_values:
            binning_values["reconstruction"] = binning_values["real"]

    print("Computing statistics (moments%s)..." % ("" if args.skip_r else " + CAS/Gini-M20/MID"))
    tables = compute_statistics(
        datasets,
        pixel_scale=args.pixel_scale,
        morph_crop=args.morph_crop,
        pool_size=args.pool_size,
        compute_morph=not args.skip_r,
    )

    for name, table in tables.items():
        path = os.path.join(args.out_dir, "catalog_%s.fits" % name)
        table.write(path, overwrite=True)
        print("Saved %s (%d rows) -> %s" % (name, len(table), path))

    print("Rendering plots to %s" % plots_dir)
    written = make_all_plots(
        tables,
        out_dir=plots_dir,
        pixel_scale=args.pixel_scale,
        binning_values=binning_values,
        binning_label=args.binning_field or "binning quantity",
        reference_name=reference_name,
        skip_morph=args.skip_r,
    )
    for path in written:
        print("  -", path)


if __name__ == "__main__":
    main()
