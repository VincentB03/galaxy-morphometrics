# galaxy-morphometrics

Computes morphometric statistics on galaxy postage stamps and compares their
distributions across datasets: real images, their autoencoder reconstructions,
and samples from a latent normalizing flow.

| Statistic | Reference |
|---|---|
| HSM adaptive moments (GalSim): size `sigma_e`, ellipticity `e`/`g`, `rho4`, flux `amp` | Hirata & Seljak (2003), Mandelbaum et al. (2005) |
| CAS: Concentration, Asymmetry | Conselice (2003) |
| Gini / M20 | Lotz, Primack & Madau (2004) |
| MID: Multimode, Intensity, Deviation | Freeman et al. (2013) |

Statistics and plots are adapted from `deepgal/validation` in
[deep_galaxy_models](https://github.com/McWilliamsCenter/deep_galaxy_models)
([Lanusse et al. 2020](https://arxiv.org/abs/2008.03833)), generalized to any
Hugging Face dataset and any number of named datasets.

## Installation

```bash
pip install -r requirements.txt
```

If `pip install galsim` fails, see the
[GalSim install instructions](https://github.com/GalSim-developers/GalSim).

**R backend** (CAS, Gini-M20, MID, called through `rpy2`). Not needed with `--skip-r`.

```bash
conda install -c conda-forge r-base
R -e 'install.packages("R.utils")'
# SDMTools is archived on CRAN: build it from source
wget -q https://cran.r-project.org/src/contrib/Archive/SDMTools/SDMTools_1.1-221.2.tar.gz
tar xzf SDMTools_1.1-221.2.tar.gz
# Only needed with R >= 4 (e.g. Colab), where the build fails with "'PI' undeclared":
sed -i '9a #include <math.h>\n#define PI M_PI' SDMTools/src/pointinpolygon.c
sed -i '7a #define PI M_PI' SDMTools/src/vincenty.geodesics.c
R CMD INSTALL SDMTools
```

**Autoencoder and flow** (`WandBGalaxyAutoencoder`, `--flow-run`). They need the
`pshear` package from [Train-AE](https://github.com/VincentB03/Train-AE), which
is not pip-installable:

```bash
git clone https://github.com/VincentB03/Train-AE.git
export PYTHONPATH="$PWD/Train-AE:$PYTHONPATH"
pip install -r Train-AE/requirements.txt   # install jax first, with the right CUDA build
```

**Private Hugging Face datasets**: `export HF_TOKEN=hf_...` (or `--hf-token`).

## Usage

```bash
python run_morphometrics.py \
    --dataset your-org/your-dataset --image-field sci_subtracted \
    --psf-field psf_stamp --noise-map-field noise_map --mask-field binary_mask \
    --n-samples 2000 --stamp-size 64 \
    --autoencoder galmorph.autoencoder:WandBGalaxyAutoencoder \
    --encoder-path entity/project/ae_run_id --decoder-path 1400 \
    --flow-run entity/project/flow_run_id --flow-epoch 500 \
    --out-dir results
```

This writes one catalog per dataset to `results/catalog_<name>.fits` and all
plots to `results/plots/`. The datasets are:

| Name | Content | Enabled by |
|---|---|---|
| `real` | stamps as loaded | always |
| `reconstruction` | encode → decode → reconvolution with the object's own PSF | `--autoencoder` |
| `flow_prior` | z ~ flow, decoded by the same autoencoder, reconvolved with a PSF drawn from the real set | `--flow-run` |

`flow_prior` has no real counterpart object, so it only appears in the
distribution plots, not in the per-object error plots.

Main options (`--help` for the full list):

| Option | Role |
|---|---|
| `--n-samples` | Takes the **first** N rows of the split (a shuffled sample with `--streaming`), so every run sees the same objects. |
| `--test-size 0.1` | Measures only Train-AE's held-out set: `train_test_split(test_size=0.1, seed=42)["test"]` (seed set by `--split-seed`). This split is shuffled, so it is not the same as `train[90%:]`. Not compatible with `--streaming`. |
| `--psf-field` | Per-object PSF, kept at its native size. Required by `WandBGalaxyAutoencoder` and `--flow-run`. |
| `--noise-map-field` | Adds white noise scaled by the noise map to the reconstructions and flow samples, which are noise-free otherwise. The R indicators estimate their threshold and S/N from the background, so they need realistic noise. |
| `--mask-field` | Validity mask (1 = valid, 0 = bad). Bad pixels are excluded from the HSM fit and replaced by a local median for the R indicators. Stamps with more than 10 % bad pixels are skipped by the R indicators. |
| `--noise-seed`, `--flow-seed`, `--psf-seed` | `--noise-seed` seeds the reconstruction noise. `--flow-seed` seeds the flow draw, its PSF assignment and its noise. `--psf-seed` changes only the PSF assignment, to isolate its effect. |
| `--binning-field` | Catalog column (e.g. magnitude) for the binned ellipticity and `rho4` plots. |
| `--morph-crop`, `--pool-size`, `--skip-r` | Crop before the R indicators, number of worker processes, skip the R indicators. |

### Custom autoencoder

Subclass `Autoencoder` and pass it with `--autoencoder module:Class`:

```python
from galmorph.autoencoder import Autoencoder

class MyAutoencoder(Autoencoder):
    def encode(self, images):  # (N, H, W) -> (N, latent_dim)
        ...

    def decode(self, codes):   # (N, latent_dim) -> (N, H, W)
        ...
```

`--encoder-path` and `--decoder-path` are passed to the constructor as
positional arguments. For `WandBGalaxyAutoencoder` they are the WandB run path
and the checkpoint epoch.

### As a library

```python
from galmorph.pipeline import compute_statistics
from galmorph.plotting import make_all_plots

tables = compute_statistics({"real": real, "reconstruction": recon}, pixel_scale=0.1)
make_all_plots(tables, out_dir="plots", reference_name="real", paired_names=["reconstruction"])
```

## Notebooks

| Notebook | Content |
|---|---|
| [`real_vs_reconstruction_morphology`](examples/real_vs_reconstruction_morphology.ipynb) | Figure 4 of [Csizi et al. (2025)](https://arxiv.org/abs/2409.07528): M20, Gini, C and A, original vs reconstruction, from the FITS catalogs. |
| [`noise_ablation_test_split`](examples/noise_ablation_test_split.ipynb) | Effect of the added noise on the indicators, on the test split. |
| [`visualize_stamps`](examples/visualize_stamps.ipynb) | Raw, masked, reconstructed and noisy stamps side by side ([Colab variant](examples/visualize_stamps_colabversion.ipynb)). |

## Layout

```
run_morphometrics.py   CLI
galmorph/
  data.py              Hugging Face dataset -> postage stamps, noise
  autoencoder.py       Autoencoder interface, WandB autoencoder and flow
  stats.py             HSM moments (GalSim), CAS/Gini-M20/MID (R)
  pipeline.py          Statistics for several named datasets
  plotting.py          Comparison plots
  r_indicators/        R code from deep_galaxy_models
examples/              Notebooks
```
