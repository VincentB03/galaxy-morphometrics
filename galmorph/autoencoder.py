# Pluggable autoencoder interface used to reconstruct galaxy images before
# computing morphometric statistics on the reconstructions.
#
# Implement `encode`/`decode` for whatever model you have (a Hugging Face
# model, a TF-Hub module, a plain PyTorch checkpoint, ...) and pass an
# instance of your subclass to `run_morphometrics.py` via
# `--autoencoder module:ClassName`.
from abc import ABC, abstractmethod
import numpy as np


class Autoencoder(ABC):
    """Minimal encode/decode interface for image reconstruction."""

    @abstractmethod
    def encode(self, images):
        """images: (N, H, W) float array -> codes: (N, latent_dim) array"""
        raise NotImplementedError

    @abstractmethod
    def decode(self, codes):
        """codes: (N, latent_dim) array -> images: (N, H, W) float array"""
        raise NotImplementedError

    def reconstruct(self, images, batch_size=256):
        """Encodes then decodes `images`, batch by batch."""
        outputs = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            outputs.append(self.decode(self.encode(batch)))
        return np.concatenate(outputs, axis=0)


class IdentityAutoencoder(Autoencoder):
    """No-op autoencoder: reconstruction == input. Useful to sanity-check
    the pipeline (real vs real should give near-perfect agreement) or as a
    placeholder while you wire up a real model."""

    def encode(self, images):
        return images

    def decode(self, codes):
        return codes


class TFHubVAEAutoencoder(Autoencoder):
    """
    Wraps a pair of TF1-style TF-Hub modules exposing an `encoder` module
    (with a "sample" output, as produced by `deepgal.VAEEstimator`) and a
    `decoder` module (plain image output). This matches the layout of
    modules/vae_16/{encoder,decoder} in the deep_galaxy_models repo.

    Requires `tensorflow` (1.x, or 2.x with `tf.compat.v1`) and
    `tensorflow_hub`.
    """

    def __init__(self, encoder_path, decoder_path):
        import tensorflow.compat.v1 as tf
        import tensorflow_hub as hub

        tf.disable_eager_execution()
        self._tf = tf

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.encoder = hub.Module(encoder_path)
            self.decoder = hub.Module(decoder_path)

            self._input_ph = tf.placeholder(tf.float32, shape=[None, None, None, 1])
            self._code = self.encoder(self._input_ph, as_dict=True)["sample"]

            self._code_ph = tf.placeholder(tf.float32, shape=[None, self._code.shape[-1]])
            self._recon = self.decoder(self._code_ph)

            self.session = tf.Session(graph=self.graph)
            self.session.run(tf.global_variables_initializer())
            self.session.run(tf.tables_initializer())

    def encode(self, images):
        images = np.asarray(images)[..., None].astype(np.float32)
        return self.session.run(self._code, feed_dict={self._input_ph: images})

    def decode(self, codes):
        recon = self.session.run(self._recon, feed_dict={self._code_ph: codes})
        return np.squeeze(recon, axis=-1)


def _unwrap_wandb_config(cfg):
    """Flattens wandb's {desc: null, value: X} config format to X per key,
    and coerces string-encoded literals (kernel_size, nested dict keys)
    back to their Python types."""
    import ast

    result = {}
    for k, v in cfg.items():
        if isinstance(v, dict) and set(v.keys()) <= {"desc", "value"}:
            v = v["value"]
        if isinstance(v, dict):
            try:
                v = {int(kk): vv for kk, vv in v.items()}
            except (ValueError, TypeError):
                pass
        if isinstance(v, str):
            try:
                v = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                pass
        result[k] = v
    return result


def _fetch_wandb_checkpoint(run_path, epoch, cache_dir):
    """
    Downloads (and caches under `cache_dir`) a run's config.yaml and
    model_checkpoint_<epoch>.eqx from Weights & Biases -- the layout
    written by both `pshear.utils.dump_galaxy_autoencoder` and `dump_flow`
    -- and returns the local `epoch_dir` containing both files, patched so
    `pshear.utils.load_galaxy_autoencoder`/`load_flow` can read it
    directly.

    If both files are already cached (e.g. pre-downloaded on a machine
    with internet access and copied over), the WandB API is skipped
    entirely, so this works on a compute node with no network access at
    all.
    """
    import shutil
    from pathlib import Path

    import yaml

    epoch = int(epoch)
    checkpoint_fname = "model_checkpoint_%d.eqx" % epoch
    run_root_dir = Path(cache_dir) / run_path.rsplit("/", 1)[-1]
    epoch_dir = run_root_dir / ("epoch_%d" % epoch)
    epoch_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_root_dir / "config.yaml"
    checkpoint_path = epoch_dir / checkpoint_fname

    if not (config_path.exists() and checkpoint_path.exists()):
        import wandb

        api = wandb.Api()
        run = api.run(run_path)
        for file in run.files():
            remote_basename = Path(file.name).name
            if remote_basename not in {"config.yaml", checkpoint_fname}:
                continue
            dest = (
                run_root_dir / remote_basename
                if remote_basename == "config.yaml"
                else epoch_dir / remote_basename
            )
            if dest.exists():
                continue
            downloaded = file.download(root=str(dest.parent), replace=True)
            downloaded_path = Path(downloaded.name)
            if downloaded_path != dest:
                downloaded_path.rename(dest)

    missing = [p for p in (config_path, checkpoint_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing files: %s (check that run %r contains them, or that "
            "cache_dir points at a pre-populated cache when offline)"
            % (", ".join(str(p) for p in missing), run_path)
        )

    shutil.copy(config_path, epoch_dir / "config.yaml")
    patched_config_path = epoch_dir / "config.yaml"
    with open(patched_config_path) as f:
        cfg = _unwrap_wandb_config(yaml.full_load(f))
    with open(patched_config_path, "w") as f:
        yaml.dump(cfg, f)

    return epoch_dir


class WandBGalaxyAutoencoder(Autoencoder):
    """
    Loads a JAX/Equinox galaxy autoencoder (from the `pshear` package) from
    a Weights & Biases run, and reconstructs images by encoding, decoding,
    then reconvolving with each galaxy's PSF, matching the training/eval
    convention: real "sci_subtracted" stamps are only ever compared against
    PSF-reconvolved reconstructions, never against the raw intrinsic (AE
    latent-space) output.

    The model architecture itself is not specified here: `config.yaml`
    downloaded from the run fully describes it, and `pshear.utils.
    load_galaxy_autoencoder` rebuilds the equinox model from that config.

    Requires `equinox`, `jax`, `wandb`, `pyyaml` and `pshear` installed.

    `encoder_path`/`decoder_path` (the CLI's generic positional autoencoder
    args) double up here as `run_path` ("entity/project/run_id") and
    `epoch`, e.g.:

        python run_morphometrics.py \\
            --autoencoder galmorph.autoencoder:WandBGalaxyAutoencoder \\
            --encoder-path entity/project/run_id --decoder-path 1400 \\
            --psf-field psf_stamp ...

    Note `reconstruct()` requires the paired `psf` stamps (see
    `galmorph.data.load_hf_stamps`'s `psf_field` argument) — `encode`/
    `decode` alone are not meaningful without the PSF reconvolution step.
    """

    def __init__(self, run_path, epoch, cache_dir="wandb_weights", seed=42):
        import equinox as eqx
        import jax
        from pshear.utils import load_galaxy_autoencoder

        self._jax = jax
        self._jnp = jax.numpy
        self._key = jax.random.PRNGKey(seed)

        epoch = int(epoch)
        epoch_dir = _fetch_wandb_checkpoint(run_path, epoch, cache_dir)
        self.model = eqx.nn.inference_mode(load_galaxy_autoencoder(epoch_dir, epoch=epoch), True)

    def encode(self, images):
        raise NotImplementedError(
            "WandBGalaxyAutoencoder needs the paired PSF to reconvolve; call reconstruct(images, psf=...) instead"
        )

    def decode(self, codes):
        raise NotImplementedError(
            "WandBGalaxyAutoencoder needs the paired PSF to reconvolve; call reconstruct(images, psf=...) instead"
        )

    def reconstruct(self, images, psf, batch_size=256):
        images = np.asarray(images)
        psf = np.asarray(psf)
        if len(psf) != len(images):
            raise ValueError("psf and images must be the same length (%d vs %d)" % (len(psf), len(images)))

        jax, jnp = self._jax, self._jnp
        outputs = []
        for i in range(0, len(images), batch_size):
            img_batch = jnp.asarray(images[i : i + batch_size])[:, None]
            psf_batch = jnp.asarray(psf[i : i + batch_size])[:, None]
            n = img_batch.shape[0]
            self._key, subkey = jax.random.split(self._key)
            keys = jax.random.split(subkey, n)
            z = jax.vmap(self.model.encode)(img_batch, keys)
            g = jax.vmap(self.model.decode)(z, keys)
            y = jax.vmap(self.model.convolve)(g, psf_batch)
            outputs.append(np.asarray(jnp.squeeze(y, axis=1)))
        return np.concatenate(outputs, axis=0)


class WandBGalaxyFlow:
    """
    Loads a JAX/Equinox latent normalizing flow (`pshear.nn.flow`, trained
    by Train-AE's `experiments/train_flow.py`) from a Weights & Biases run,
    and draws unconditional samples from it: z ~ flow, decoded and
    reconvolved with a PSF, matching `train_flow.py`'s own `sample_images`
    sanity check. This is meant to be added as an extra named dataset (e.g.
    "flow_prior") alongside "real"/"reconstruction" in `run_morphometrics.py`,
    to check whether the flow recovers the real data's morphometric
    distributions, not to compare individual objects against it.

    The flow only makes sense in the latent space of the specific
    autoencoder it was trained against (`train_flow.py`'s `ae_run_dir`/
    `ae_epoch`), so it does not load its own decoder: pass the
    `WandBGalaxyAutoencoder` already built for the "reconstruction" dataset.

    A flow sample has no real galaxy behind it, so it has no PSF of its own
    either -- `sample()` takes one PSF stamp per requested sample, meant to
    be resampled from the real dataset's own PSFs (see `run_morphometrics.py`).

    Requires `equinox`, `jax`, `wandb`, `pyyaml` and `pshear` installed.

        python run_morphometrics.py \\
            --autoencoder galmorph.autoencoder:WandBGalaxyAutoencoder \\
            --encoder-path entity/project/ae_run_id --decoder-path 1400 \\
            --psf-field psf_stamp \\
            --flow-run entity/project/flow_run_id --flow-epoch 500 ...
    """

    def __init__(self, ae, run_path, epoch, cache_dir="wandb_weights", seed=42):
        import jax
        from pshear.utils import load_flow

        if not hasattr(ae, "model"):
            raise TypeError(
                "WandBGalaxyFlow needs the WandBGalaxyAutoencoder built for "
                "the flow's own latent space (or anything exposing the same "
                ".model), got %r" % (type(ae),)
            )

        self._jax = jax
        self._jnp = jax.numpy
        self._key = jax.random.PRNGKey(seed)
        self.ae_model = ae.model

        epoch = int(epoch)
        epoch_dir = _fetch_wandb_checkpoint(run_path, epoch, cache_dir)
        self.flow = load_flow(epoch_dir, epoch=epoch)

    def sample(self, n, psf, batch_size=256):
        """
        Draws `n` unconditional samples from the flow prior, decodes and
        reconvolves each with the corresponding row of `psf`, shape
        (n, H, W) -- e.g. resampled with replacement from the real
        dataset's PSF stamps, since flow samples don't have one of their
        own.
        """
        psf = np.asarray(psf)
        if len(psf) != n:
            raise ValueError("psf must have length n=%d, got %d" % (n, len(psf)))

        jax, jnp = self._jax, self._jnp
        outputs = []
        for i in range(0, n, batch_size):
            psf_batch = jnp.asarray(psf[i : i + batch_size])[:, None]
            b = psf_batch.shape[0]
            self._key, subkey = jax.random.split(self._key)
            z = self.flow.sample(key=subkey, sample_shape=(b,))
            z = self.flow.unflatten_latent(z)
            g = jax.vmap(self.ae_model.decode)(z)
            y = jax.vmap(self.ae_model.convolve)(g, psf_batch)
            outputs.append(np.asarray(jnp.squeeze(y, axis=1)))
        return np.concatenate(outputs, axis=0)
