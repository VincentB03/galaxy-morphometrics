# galaxy-morphometrics

Standalone toolkit to compute galaxy morphometric statistics and reproduce
the comparison plots from
[deep_galaxy_models](https://github.com/McWilliamsCenter/deep_galaxy_models)'s
`deepgal/validation` (Lanusse et al. 2020), applied to your own data instead
of the COSMOS + GalSim-Hub pipeline the original repo was built around.

Given postage-stamp galaxy images (loaded from a Hugging Face dataset),
optionally their reconstruction through a pretrained autoencoder, and
optionally unconditional samples from a latent normalizing flow trained on
that autoencoder's latent space, this repo computes:

- **HSM adaptive moments** (GalSim `FindAdaptiveMom`): size `sigma_e`,
  ellipticity `e`/`e1`/`e2`/`g`/`g1`/`g2`, `rho4`, flux `amp`.
- **CAS** (Concentration, Asymmetry) — Conselice (2003).
- **Gini / M20** — Lotz, Primack & Madau (2004).
- **MID** (Multimode, Intensity, Deviation) — Freeman et al. (2013).

...and renders the same kind of plots as the paper: ellipticity/size
distributions, `rho4` vs magnitude/size, Gini-M20, M-I, M-D, MID
distributions (one curve per named dataset — real, reconstruction,
flow_prior, or any others you add), and (when a reference/reconstruction
pair is available) per-object reconstruction error plots.

## Origin

The statistics and R routines are ported from `deepgal/validation/` in
deep_galaxy_models with minimal changes (path fixes only). The plotting
logic is generalized from `deepgal/validation/plotting.py` and the
`Figure_Moments.ipynb` / `Figure_Morphology.ipynb` notebooks to work on an
arbitrary number of named datasets instead of the hardcoded
real/mock/parametric triplet.

## Install

```bash
pip install -r requirements.txt
```

GalSim itself has non-Python dependencies (FFTW, TMV/Eigen) — follow the
[GalSim install instructions](https://github.com/GalSim-developers/GalSim)
if `pip install galsim` fails.

### Private Hugging Face datasets

To load a private or gated dataset, set the `HF_TOKEN` environment
variable to a Hugging Face access token (Settings -> Access Tokens on
huggingface.co) before running:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
python run_morphometrics.py --dataset your-org/your-private-dataset ...
```

`HF_TOKEN` is picked up automatically; alternatively pass `--hf-token` on
the CLI, or `hf_token=...` to `galmorph.data.load_hf_stamps` directly.

### R backend (CAS / Gini-M20 / MID)

These indicators call into R via `rpy2`. You need:

```bash
# R itself, e.g. via conda or your OS package manager
conda install -c conda-forge r-base

# SDMTools was archived on CRAN; install the last release from the archive.
# Its dependency R.utils isn't archived, install it from CRAN first or the
# SDMTools install will fail with "dependency 'R.utils' is not available".
R -e 'install.packages("R.utils")'
R -e 'install.packages("https://cran.r-project.org/src/contrib/Archive/SDMTools/SDMTools_1.1-221.2.tar.gz", repos=NULL, type="source")'
```

On recent toolchains (e.g. Ubuntu 22.04 / R >= 4.x, including Colab), the
build fails with `error: 'PI' undeclared` in `pointinpolygon.c` and
`vincenty.geodesics.c` — `PI` used to come in transitively via `<R.h>` /
`<Rmath.h>` and no longer does (only `M_PI` is guaranteed). Patch both
source files before installing:

```bash
cd /tmp
wget -q https://cran.r-project.org/src/contrib/Archive/SDMTools/SDMTools_1.1-221.2.tar.gz
tar xzf SDMTools_1.1-221.2.tar.gz
sed -i '9a #include <math.h>\n#define PI M_PI' SDMTools/src/pointinpolygon.c
sed -i '7a #define PI M_PI' SDMTools/src/vincenty.geodesics.c
R CMD INSTALL SDMTools
```

If you don't need CAS/Gini-M20/MID, pass `--skip-r` to skip this
dependency entirely — you still get the full set of HSM moments and
their plots.

### Autoencoder (optional)

Reconstructing images requires a model. `galmorph/autoencoder.py` defines
a minimal `Autoencoder` interface (`encode`/`decode`) plus:

- `IdentityAutoencoder`: no-op, useful to sanity-check the pipeline.
- `TFHubVAEAutoencoder`: wraps a TF1-style TF-Hub encoder/decoder module
  pair, compatible with `modules/vae_16/{encoder,decoder}` from
  deep_galaxy_models (needs `tensorflow` + `tensorflow_hub`).
- `WandBGalaxyAutoencoder`: downloads a JAX/Equinox galaxy autoencoder
  checkpoint + config from a Weights & Biases run and reconstructs images
  by encode -> decode -> **reconvolve with the object's own PSF**, matching
  the training/eval convention (real stamps are PSF-convolved, so
  reconstructions must be too before comparing statistics). Requires
  `equinox`, `jax`, `wandb`, `pyyaml` and your own `pshear` package:

  ```bash
  # pshear lives in a private repo; requires a GitHub access token
  pip install "git+https://${GITHUB_TOKEN}@github.com/VincentB03/Train-AE.git"
  # add #subdirectory=pshear if the package isn't at the repo root
  ```

  ```bash
  python run_morphometrics.py \
      --dataset your-org/your-dataset --image-field sci_subtracted \
      --psf-field psf_stamp --n-samples 2000 --stamp-size 64 \
      --autoencoder galmorph.autoencoder:WandBGalaxyAutoencoder \
      --encoder-path entity/project/run_id --decoder-path 1400 \
      --out-dir results
  ```

  (`--encoder-path`/`--decoder-path` double up as the WandB run path and
  checkpoint epoch here — see the class docstring.) `--psf-field` is
  required for this autoencoder: it loads a per-object PSF stamp
  alongside the image, kept at its native size (not resized to
  `--stamp-size` — the convolution step handles the size mismatch), which
  `WandBGalaxyAutoencoder.reconstruct` needs to reconvolve the decoded
  image before statistics are computed on it.

#### Adding noise to reconstructions

Autoencoder reconstructions come out noise-free, but the CAS/Gini-M20/MID
indicators (`galmorph/r_indicators/`) estimate their segmentation
threshold and S/N from the background pixel scatter — degenerate on a
noise-free image, and not comparable to the real images' own S/N. Pass
`--noise-map-field` to add a white-noise realization, scaled by a
per-object noise map column from the dataset (per-pixel noise standard
deviation, fit to `--stamp-size` like `--image-field`), to the
reconstruction before statistics are computed on it:

```bash
python run_morphometrics.py \
    --dataset your-org/your-dataset --image-field sci_subtracted \
    --psf-field psf_stamp --noise-map-field noise_map \
    --n-samples 2000 --stamp-size 64 \
    --autoencoder galmorph.autoencoder:WandBGalaxyAutoencoder \
    --encoder-path entity/project/run_id --decoder-path 1400 \
    --out-dir results
```

`--noise-seed` (default `0`) seeds the noise draw for reproducibility. The
"real" images are left untouched — they already carry their own noise.
Use `galmorph.data.add_noise(images, noise_map, seed=...)` directly if
you're calling the library instead of the CLI.

#### Flow prior sampling (optional)

A latent normalizing flow fit to `WandBGalaxyAutoencoder`'s latent space
(Train-AE's `experiments/train_flow.py`) should, if it has learned that
space correctly, reproduce the real data's morphometric distribution when
its samples are decoded. Pass `--flow-run` to check this: it adds a third
`flow_prior` dataset — unconditional samples z ~ flow, decoded through the
*same* autoencoder as `--autoencoder` and reconvolved with PSFs resampled
from the real dataset (a flow sample has no real galaxy, and therefore no
PSF, of its own) — so every distribution plot shows real vs reconstruction
vs flow_prior. The per-object reconstruction-error plots are unaffected:
`flow_prior` isn't index-aligned with "real" and is never included there.

```bash
python run_morphometrics.py \
    --dataset your-org/your-dataset --image-field sci_subtracted \
    --psf-field psf_stamp --n-samples 2000 --stamp-size 64 \
    --autoencoder galmorph.autoencoder:WandBGalaxyAutoencoder \
    --encoder-path entity/project/ae_run_id --decoder-path 1400 \
    --flow-run entity/project/flow_run_id --flow-epoch 500 \
    --out-dir results
```

Requires `--autoencoder galmorph.autoencoder:WandBGalaxyAutoencoder` and
`--psf-field`. `--flow-n-samples` defaults to the real dataset's own count;
`--flow-seed` (default `0`) seeds both the flow draw and the PSF
resampling. Omit `--flow-run` to skip this curve entirely — everything
else behaves exactly as before.

To plug in your own pretrained model (a Hugging Face model, a PyTorch
checkpoint, etc.), subclass `Autoencoder` in a small module of your own and
point `--autoencoder` at it, e.g.:

```python
# my_autoencoder.py
from galmorph.autoencoder import Autoencoder

class MyAutoencoder(Autoencoder):
    def encode(self, images):
        ...  # images: (N, H, W) float array -> (N, latent_dim)

    def decode(self, codes):
        ...  # (N, latent_dim) -> (N, H, W) float array
```

```bash
python run_morphometrics.py --dataset ... --autoencoder my_autoencoder:MyAutoencoder
```

## Usage

```bash
python run_morphometrics.py \
    --dataset your-org/your-galaxy-dataset --split train \
    --image-field image --n-samples 2000 --stamp-size 128 \
    --autoencoder galmorph.autoencoder:TFHubVAEAutoencoder \
    --encoder-path modules/vae_16/encoder --decoder-path modules/vae_16/decoder \
    --out-dir results
```

This will:

1. Load `n-samples` images from the dataset and fit them to `stamp-size` x
   `stamp-size` postage stamps.
2. Reconstruct them with the autoencoder (skip `--autoencoder` to only
   analyze the real images).
3. Compute HSM moments and (unless `--skip-r`) CAS/Gini-M20/MID for every
   named set of images.
4. Save one FITS catalog per set to `results/catalog_<name>.fits`.
5. Render every applicable plot to `results/plots/`.

Run `python run_morphometrics.py --help` for the full list of options
(pixel scale, morphology crop size, worker pool size, an optional
`--binning-field` catalog column for the magnitude/size-binned plots, ...).

### Using the library directly

For more control (e.g. comparing more than two datasets, or data that
isn't on the Hugging Face Hub), call the pieces directly:

```python
from galmorph.pipeline import compute_statistics
from galmorph.plotting import make_all_plots

datasets = {"real": real_stamps, "reconstruction": recon_stamps, "flow_prior": flow_stamps}
tables = compute_statistics(datasets, pixel_scale=0.03, morph_crop=64)
make_all_plots(
    tables, out_dir="results/plots",
    reference_name="real", paired_names=["reconstruction"],  # flow_prior isn't index-aligned with real
)
```

## Repository layout

```
run_morphometrics.py   CLI entry point
galmorph/
  data.py              Hugging Face dataset -> postage stamps
  autoencoder.py       Autoencoder interface + TF-Hub / identity / WandB (AE, flow) implementations
  stats.py             HSM moments (GalSim) + CAS/Gini-M20/MID (R via rpy2)
  pipeline.py          Per-dataset / multi-dataset statistics computation
  plotting.py          All comparison plots
  r_indicators/        R implementation of CAS/Gini-M20/MID (ported as-is)
```
