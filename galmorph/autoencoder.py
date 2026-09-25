# Autoencoders used to build the "reconstruction" dataset, and the latent flow
# used for "flow_prior". Plug in your own model by subclassing `Autoencoder` and
# passing `--autoencoder module:ClassName` to run_morphometrics.py.
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
    """No-op autoencoder (reconstruction == input), to sanity-check the pipeline."""

    def encode(self, images):
        return images

    def decode(self, codes):
        return codes


def _unwrap_wandb_config(cfg):
    """Flattens WandB's {desc, value} config entries and parses string-encoded
    literals and integer dict keys back to Python types."""
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
    Downloads a WandB run's config.yaml and model_checkpoint_<epoch>.eqx into
    `cache_dir` and returns the epoch directory, in the layout read by
    `pshear.utils.load_galaxy_autoencoder`/`load_flow`. Cached files are
    reused without calling the WandB API, so this also works offline.
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
    JAX/Equinox galaxy autoencoder from `pshear`, loaded from a WandB run (the
    architecture is rebuilt from the run's config.yaml).

    Reconstructs by encoding, decoding, then reconvolving with each object's
    PSF, since real stamps are PSF-convolved. `reconstruct` therefore requires
    `psf`; `encode`/`decode` alone are not supported.

    `run_path` ("entity/project/run_id") and `epoch` map to the CLI's
    `--encoder-path` and `--decoder-path`.
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
    Latent normalizing flow from `pshear` (trained by Train-AE's
    `experiments/train_flow.py`), loaded from a WandB run. Draws unconditional
    samples z ~ flow and decodes them with `ae`, which must be the
    `WandBGalaxyAutoencoder` whose latent space the flow was trained on.
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
        self._key = jax.random.key(seed)
        self.ae_model = ae.model

        epoch = int(epoch)
        epoch_dir = _fetch_wandb_checkpoint(run_path, epoch, cache_dir)
        self.flow = load_flow(epoch_dir, epoch=epoch)

    def sample(self, n, psf, batch_size=256):
        """Draws `n` samples, each decoded and reconvolved with the matching row
        of `psf`, shape (n, H, W)."""
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
