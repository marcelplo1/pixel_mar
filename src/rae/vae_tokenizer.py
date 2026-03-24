import torch
import torch.nn as nn
from math import sqrt

from rae.vae import AutoencoderKL


class VaeTokenizer(nn.Module):
    """
    VAE Tokenizer: frozen KL-VAE encoder + decoder from LDM.
    """
    def __init__(
        self,
        vae_path='pretrained_models/vae/kl16.ckpt',
        embed_dim=16,
        ch_mult=(1, 1, 2, 2, 4),
        latent_scale=0.2325,
        img_size=256,
    ):
        super().__init__()

        self.vae = AutoencoderKL(
            embed_dim=embed_dim,
            ch_mult=ch_mult,
            ckpt_path=vae_path,
        )
        self.vae.requires_grad_(False)

        self.latent_dim = embed_dim       
        self.latent_scale = latent_scale  # From LDM
        self.vae_stride = 16             # spatial downsampling factor
        self.num_patches = (img_size // self.vae_stride) ** 2  # 256 for 256x256

    @torch.no_grad()
    def encode(self, images):
        """
        Encode images to normalized VAE latent tokens.
        images: [B, 3, H, W] in [-1, 1]
        returns: [B, N, D] where N = (H/16)*(W/16), D = 16
        """
        posterior = self.vae.encode(images)
        z = posterior.mode()
        z = z * self.latent_scale

        B, D, H, W = z.shape
        N = H * W
        z = z.reshape(B, D, N).permute(0, 2, 1)
        return z

    @torch.no_grad()
    def decode(self, z):
        """
        Decode normalized VAE latent tokens back to images.
        z: [B, N, D]
        returns: [B, 3, H, W] in [-1, 1]
        """
        B, N, D = z.shape
        H = W = int(sqrt(N))
        z = z.permute(0, 2, 1).reshape(B, D, H, W)
        z = z / self.latent_scale
        images = self.vae.decode(z)
        return images