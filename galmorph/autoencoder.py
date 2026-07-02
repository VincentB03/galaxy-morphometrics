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
