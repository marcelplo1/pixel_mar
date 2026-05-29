import torch
import torch.nn as nn
from transformers import ViTMAEForPreTraining, AutoImageProcessor
from math import sqrt
from image_tokenizers.rae_decoder import GeneralDecoder
from transformers import AutoConfig


class MaeTokenizer(nn.Module):
    """
        MAE Tokenizer: frozen MAE encoder + frozen RAE decoder.
    """
    def __init__(
        self,
        mae_path='facebook/vit-mae-base',
        rae_decoder_config_path='./src/tokenizers/rae_decoder_configs/ViTXL',
        encoder_input_size=256,
        decoder_patch_size=16,
        eps=1e-5,
        rae_decoder_ckp="./tokenizer_models/mae/model.pt",
        rae_norm_stats="./tokenizer_models/mae/stat.pt"
    ):
        super().__init__()

        # ---- Encoder (frozen MAE) ----
        self.encoder = ViTMAEForPreTraining.from_pretrained(mae_path).vit
        self.encoder.requires_grad_(False)
        # Remove affine of final layernorm (match RAE training)
        self.encoder.layernorm.elementwise_affine = False
        self.encoder.layernorm.weight = None
        self.encoder.layernorm.bias = None
        # Disable masking
        self.encoder.config.mask_ratio = 0.0

        self.encoder_input_size = encoder_input_size
        self.latent_dim = self.encoder.config.hidden_size
        self.encoder_patch_size = self.encoder.config.patch_size
        self.num_patches = (encoder_input_size // self.encoder_patch_size) ** 2

        # MAE normalization (ImageNet stats from processor)
        proc = AutoImageProcessor.from_pretrained(mae_path)
        self.register_buffer('encoder_mean', torch.tensor(proc.image_mean).view(1, 3, 1, 1))
        self.register_buffer('encoder_std', torch.tensor(proc.image_std).view(1, 3, 1, 1))

        # ---- Decoder (frozen RAE decoder) ----
        self.decoder = None
        self.decoder_patch_size = decoder_patch_size
        if rae_decoder_config_path is not None:
            decoder_config = AutoConfig.from_pretrained(rae_decoder_config_path)
            decoder_config.hidden_size = self.latent_dim
            decoder_config.patch_size = decoder_patch_size
            decoder_config.image_size = int(decoder_patch_size * sqrt(self.num_patches))

            self.decoder = GeneralDecoder(decoder_config, num_patches=self.num_patches)

            if rae_decoder_ckp is not None:
                print(f"Loading pretrained RAE decoder from {rae_decoder_ckp}")
                state_dict = torch.load(rae_decoder_ckp, map_location='cpu', weights_only=False)
                keys = self.decoder.load_state_dict(state_dict, strict=False)
                if keys.missing_keys:
                    print(f"  Missing keys: {keys.missing_keys}")
                if keys.unexpected_keys:
                    print(f"  Unexpected keys: {keys.unexpected_keys}")

            self.decoder.requires_grad_(False)

        # ---- Latent normalization (precomputed stats) ----
        self.eps = eps
        if rae_norm_stats is not None:
            stats = torch.load(rae_norm_stats, map_location='cpu', weights_only=False)
            latent_mean = stats.get('mean', None)
            latent_var = stats.get('var', None)
            self.register_buffer('latent_mean', latent_mean)
            self.register_buffer('latent_var', latent_var)
            self.do_normalization = latent_var is not None
            print(f"Loaded latent normalization stats from {rae_norm_stats} "
                  f"(mean={'None' if latent_mean is None else latent_mean.shape}, "
                  f"var={'None' if latent_var is None else latent_var.shape})")
        else:
            self.do_normalization = False

    @torch.no_grad()
    def encode(self, images):
        """
        Encode images to normalized MAE feature tokens.
        """
        # Convert from [-1, 1] to [0, 1]
        x = (images + 1) / 2

        # Resize to encoder input size if needed
        _, _, h, w = x.shape
        if h != self.encoder_input_size or w != self.encoder_input_size:
            x = nn.functional.interpolate(
                x, size=(self.encoder_input_size, self.encoder_input_size),
                mode='bicubic', align_corners=False
            )

        # Apply ImageNet normalization
        x = (x - self.encoder_mean) / self.encoder_std

        # Ordered noise to prevent any masking
        patch_num = (x.shape[2] // self.encoder_patch_size) * (x.shape[3] // self.encoder_patch_size)
        noise = torch.arange(patch_num).unsqueeze(0).expand(x.shape[0], -1).to(x.device).float()

        outputs = self.encoder(x, noise, interpolate_pos_encoding=True)
        z = outputs.last_hidden_state[:, 1:]  # remove CLS token

        if self.do_normalization:
            B, N, D = z.shape
            H = W = int(sqrt(N))
            z = z.transpose(1, 2).view(B, D, H, W)
            latent_mean = self.latent_mean if self.latent_mean is not None else 0
            latent_var = self.latent_var if self.latent_var is not None else 1
            z = (z - latent_mean) / torch.sqrt(latent_var + self.eps)
            z = z.view(B, D, N).transpose(1, 2)

        return z

    @torch.no_grad()
    def decode(self, z):
        """
        Decode normalized MAE features back to pixel images.
        """
        # Denormalize (reshape to 2D to match RAE convention)
        if self.do_normalization:
            B, N, D = z.shape
            H = W = int(sqrt(N))
            z_2d = z.transpose(1, 2).view(B, D, H, W)
            latent_mean = self.latent_mean if self.latent_mean is not None else 0
            latent_var = self.latent_var if self.latent_var is not None else 1
            z_2d = z_2d * torch.sqrt(latent_var + self.eps) + latent_mean
            z = z_2d.view(B, D, N).transpose(1, 2)

        output = self.decoder(z, drop_cls_token=False).logits
        x_rec = self.decoder.unpatchify(output)

        # Un-normalize from ImageNet space to [0, 1]
        x_rec = x_rec * self.encoder_std + self.encoder_mean

        # Convert to [-1, 1] (matching dataset normalization)
        x_rec = x_rec * 2 - 1

        return x_rec