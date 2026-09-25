#!/usr/bin/env python
"""
Computes galaxy morphometric statistics (HSM moments, CAS, Gini-M20, MID) on
real postage stamps and, optionally, on their autoencoder reconstructions and
latent-flow samples, then saves one FITS catalog per dataset and the comparison
plots.

Example:
    python run_morphometrics.py \\
        --dataset your-org/your-dataset --image-field sci_subtracted \\
        --psf-field psf_stamp --n-samples 2000 --stamp-size 64 \\
        --autoencoder galmorph.autoencoder:WandBGalaxyAutoencoder \\
        --encoder-path entity/project/ae_run_id --decoder-path 1400 \\
        --flow-run entity/project/flow_run_id --flow-epoch 500 \\
        --out-dir results
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
    g_data.add_argument(
        "--test-size", type=float, default=None,
        help="Keep only the 'test' part of train_test_split(test_size=..., seed=--split-seed). "
             "0.1 gives Train-AE's held-out set (shuffled, not the same as 'train[90%%:]').",
    )
    g_data.add_argument("--split-seed", type=int, default=42, help="Seed for --test-size (Train-AE uses 42)")
    g_data.add_argument("--image-field", default="image")
    g_data.add_argument("--n-samples", type=int, default=2000)
    g_data.add_argument("--stamp-size", type=int, default=64)
    g_data.add_argument("--pixel-scale", type=float, default=0.1, help="arcsec/pixel")
    g_data.add_argument("--streaming", action="store_true")
    g_data.add_argument(
        "--binning-field", default=None,
        help="Numeric catalog column (e.g. magnitude) for the binned ellipticity/rho4 plots",
    )
    g_data.add_argument(
        "--hf-token", default=os.environ.get("HF_TOKEN"),
        help="Token for private/gated datasets (default: $HF_TOKEN)",
    )
    g_data.add_argument(
        "--psf-field", default=None,
        help="Per-object PSF column, kept at its native size. Required by "
             "WandBGalaxyAutoencoder and --flow-run.",
    )
    g_data.add_argument(
        "--noise-map-field", default=None,
        help="Per-pixel noise std column. White noise scaled by it is added to the "
             "(noise-free) reconstructions and flow samples.",
    )
    g_data.add_argument(
        "--noise-seed", type=int, default=0,
        help="Seed for the noise added to the reconstructions",
    )
    g_data.add_argument(
        "--mask-field", default=None,
        help="Per-pixel validity mask column (1 = valid, 0 = bad). Bad pixels are "
             "excluded from the HSM fit and filled in for the R indicators.",
    )

    g_ae = p.add_argument_group("autoencoder (optional)")
    g_ae.add_argument(
        "--autoencoder", default=None,
        help="'module:Class' of an Autoencoder subclass, e.g. galmorph.autoencoder:WandBGalaxyAutoencoder",
    )
    g_ae.add_argument("--encoder-path", default=None, help="First constructor arg (WandB run path for WandBGalaxyAutoencoder)")
    g_ae.add_argument("--decoder-path", default=None, help="Second constructor arg (checkpoint epoch for WandBGalaxyAutoencoder)")

    g_flow = p.add_argument_group(
        "flow prior (optional, requires WandBGalaxyAutoencoder and --psf-field)"
    )
    g_flow.add_argument(
        "--flow-run", default=None,
        help="WandB 'entity/project/run_id' of a latent flow trained on --autoencoder's latent "
             "space. Adds a 'flow_prior' dataset: z ~ flow, decoded by the same autoencoder and "
             "reconvolved with PSFs drawn from the real dataset.",
    )
    g_flow.add_argument("--flow-epoch", type=int, default=None, help="Checkpoint epoch for --flow-run")
    g_flow.add_argument(
        "--flow-cache-dir", default="wandb_weights",
        help="Local cache for the downloaded flow weights",
    )
    g_flow.add_argument(
        "--flow-n-samples", type=int, default=None,
        help="Number of flow samples (default: number of real stamps)",
    )
    g_flow.add_argument("--flow-seed", type=int, default=0, help="Seed for the flow draw, its PSF assignment and its noise")
    g_flow.add_argument(
        "--psf-seed", type=int, default=None,
        help="Seed for the PSF assignment only (default: --flow-seed), to isolate its effect",
    )

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
        mask_field=args.mask_field,
        test_size=args.test_size,
        split_seed=args.split_seed,
    )
    if extra_fields or args.psf_field or args.noise_map_field or args.mask_field:
        real_images, extra = loaded
        binning_values = {"real": extra[args.binning_field]} if args.binning_field else None
        psf_images = extra["psf"] if args.psf_field else None
        noise_map = extra["noise_map"] if args.noise_map_field else None
        real_masks = extra["mask"] if args.mask_field else None
    else:
        real_images = loaded
        binning_values = None
        psf_images = None
        noise_map = None
        real_masks = None
    print("Loaded %d postage stamps of size %dx%d" % (len(real_images), args.stamp_size, args.stamp_size))

    datasets = {"real": real_images}
    masks = {"real": real_masks} if real_masks is not None else None
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

    if args.flow_run:
        if not args.autoencoder:
            raise ValueError(
                "--flow-run requires --autoencoder galmorph.autoencoder:WandBGalaxyAutoencoder "
                "(the flow needs that autoencoder's frozen decoder for its own latent space)"
            )
        if psf_images is None:
            raise ValueError("--flow-run requires --psf-field (flow samples need a PSF to be reconvolved with)")

        print("Sampling from flow prior %s (epoch %s)" % (args.flow_run, args.flow_epoch))
        from galmorph.autoencoder import WandBGalaxyFlow

        flow_sampler = WandBGalaxyFlow(
            ae, args.flow_run, args.flow_epoch, cache_dir=args.flow_cache_dir, seed=args.flow_seed
        )
        n_flow = args.flow_n_samples or len(real_images)
        # flow samples have no PSF of their own: borrow random ones from the real dataset
        psf_seed = args.psf_seed if args.psf_seed is not None else args.flow_seed
        rng = np.random.default_rng(psf_seed)
        flow_idx = rng.integers(0, len(psf_images), size=n_flow)
        datasets["flow_prior"] = flow_sampler.sample(n_flow, psf_images[flow_idx])
        if noise_map is not None:
            print("Adding white noise scaled by --noise-map-field to the flow samples")
            datasets["flow_prior"] = add_noise(datasets["flow_prior"], noise_map[flow_idx], seed=args.flow_seed)

    # only "reconstruction" is object-by-object aligned with "real"
    paired_names = ["reconstruction"] if "reconstruction" in datasets else None

    print("Computing statistics (moments%s)..." % ("" if args.skip_r else " + CAS/Gini-M20/MID"))
    tables = compute_statistics(
        datasets,
        pixel_scale=args.pixel_scale,
        morph_crop=args.morph_crop,
        pool_size=args.pool_size,
        compute_morph=not args.skip_r,
        masks=masks,
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
        paired_names=paired_names,
        skip_morph=args.skip_r,
    )
    for path in written:
        print("  -", path)


if __name__ == "__main__":
    main()
