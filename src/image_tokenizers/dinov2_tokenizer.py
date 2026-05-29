import torch
import torch.nn as nn
from transformers import Dinov2WithRegistersModel, AutoConfig, AutoImageProcessor
from math import sqrt
from image_tokenizers.rae_decoder import GeneralDecoder


class Dinov2Tokenizer(nn.Module):
    """
        DinoV2 Tokenizer: frozen DINOv2 encoder + frozen RAE decoder.
    """
    def __init__(
        self,
        dinov2_path='facebook/dinov2-with-registers-base',
        rae_decoder_config_path='./src/tokenizers/rae_decoder_configs/ViTXL',
        encoder_input_size=224,
        decoder_patch_size=16,
        eps=1e-5,
        rae_decoder_ckp="./tokenizer_models/rae_decoder/model.pt",
        rae_norm_stats="./tokenizer_models/rae_stats/stat.pt"
    ):
        super().__init__()

        # ---- Encoder (frozen DINOv2) ----
        self.encoder = Dinov2WithRegistersModel.from_pretrained(dinov2_path)
        self.encoder.requires_grad_(False)
        self.encoder.layernorm.elementwise_affine = False
        self.encoder.layernorm.weight = None
        self.encoder.layernorm.bias = None

        self.encoder_input_size = encoder_input_size
        self.latent_dim = self.encoder.config.hidden_size
        self.encoder_patch_size = self.encoder.config.patch_size
        self.num_patches = (encoder_input_size // self.encoder_patch_size) ** 2

        # DINOv2 normalization (ImageNet stats from processor)
        proc = AutoImageProcessor.from_pretrained(dinov2_path)
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
        Encode images to normalized DINOv2 feature tokens.
        """
        # Convert from [-1, 1] to [0, 1]
        x = (images + 1) / 2

        # Resize to DINOv2 input size if needed
        _, _, h, w = x.shape
        if h != self.encoder_input_size or w != self.encoder_input_size:
            x = nn.functional.interpolate(
                x, size=(self.encoder_input_size, self.encoder_input_size),
                mode='bicubic', align_corners=False
            )

        # Apply DINOv2 normalization (ImageNet mean/std)
        x = (x - self.encoder_mean) / self.encoder_std

        outputs = self.encoder(x, output_hidden_states=True)
        unused_token_num = 5  # 1 CLS + 4 register tokens
        z = outputs.last_hidden_state[:, unused_token_num:]

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
        Decode normalized DINOv2 features back to pixel images.
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

        # Un-normalize from DINOv2 space to [0, 1]
        x_rec = x_rec * self.encoder_std + self.encoder_mean

        # Convert to [-1, 1] (matching dataset normalization)
        x_rec = x_rec * 2 - 1

        return x_rec