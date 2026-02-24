import numpy as np
from model.model_utils import get_2d_sincos_pos_embed
from scipy import stats
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from transformers import ViTMAEConfig, ViTMAEForPreTraining


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class MAE(nn.Module):
    def __init__(
            self,
            img_size,
            patch_size=16,
            channels=3,
            num_classes=10,
            ema_decay=0.9999,
            encoder_dim=768,
            decoder_dim=768,
            encoder_depth=12,
            decoder_depth=12,
            encoder_num_heads=12,
            decoder_num_heads=12,
            mlp_ratio=4.0,
            dropout=0.1,
            buffer_size=64,
            min_mask_rate = 0.7,
            lable_dropout = 0.1,
            mae_config = None
        ):
        super().__init__()

        self.patch_size = patch_size
        self.min_mask_rate = min_mask_rate
        self.buffer_size = buffer_size
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.img_size = img_size
        self.channels = channels
        self.seq_len = (img_size // patch_size) ** 2
        self.embed_dim = channels * patch_size**2

        self.mask_token  = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.class_emb = nn.Embedding(num_classes, encoder_dim)

        self.decoder_embed = nn.Linear(self.encoder_dim, self.decoder_dim, bias=True)
        self.decoder_pos_emb = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, decoder_dim),  requires_grad=True)
        self.decoder_block = nn.ModuleList([
            Block(decoder_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm,
                  proj_drop=dropout, attn_drop=dropout) for _ in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(decoder_dim, eps=1e-6)

        self.label_drop_prob = lable_dropout
        self.fake_latent = nn.Parameter(torch.zeros(1, encoder_dim))

        self.ema_decay=ema_decay
        self.ema_params = None

        self.initialize_weights()

        # Frozen pretrained encoder (after initialize_weights to avoid overwriting)
        config = ViTMAEConfig.from_pretrained(mae_config)
        config.patch_size = int(patch_size)
        config.num_channels = int(channels)
        self.model_name = mae_config
        self.encoder_mae = ViTMAEForPreTraining.from_pretrained(self.model_name, config=config).vit
        for param in self.encoder_mae.parameters():
            param.requires_grad = False
        self.encoder_mae.eval()

        # ImageNet normalization
        self.register_buffer('encoder_mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('encoder_std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def initialize_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

        grid_size = int(self.seq_len ** 0.5)
        pos_embed_grid = get_2d_sincos_pos_embed(self.decoder_dim, grid_size)
        full_pos_embed = torch.zeros(self.seq_len + self.buffer_size, self.decoder_dim)
        full_pos_embed[self.buffer_size:, :] = torch.from_numpy(pos_embed_grid).float()
        self.decoder_pos_emb.data.copy_(full_pos_embed.unsqueeze(0))

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.class_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.fake_latent, std=.02)

    def forward_encoder_train(self, x):
        """Training: HF encoder with built-in random masking."""
        mask_rate = stats.truncnorm((self.min_mask_rate - 1.0) / 0.25, 0, loc=1.0, scale=0.25).rvs(1)[0]
        self.encoder_mae.config.mask_ratio = mask_rate
        output = self.encoder_mae(x, interpolate_pos_encoding=True)
        return output.last_hidden_state, output.mask, output.ids_restore

    def encoder_generate(self, x, mask_orders, num_visible):
        """Generation: Embeddings for all patches, then encode only visible ones."""
        self.encoder_mae.config.mask_ratio = 0
        noise = torch.arange(self.seq_len).unsqueeze(0).expand(x.shape[0], -1).to(x.device).float()
        x_embedding = self.encoder_mae.embeddings(x, noise=noise, interpolate_pos_encoding=True)
        x_embedding = x_embedding[0] if isinstance(x_embedding, tuple) else x_embedding

        B, N, D = x_embedding.shape
        cls_token = x_embedding[:, :1, :]
        img_tokens = x_embedding[:, 1:, :]

        visible = torch.gather(img_tokens, dim=1, index=mask_orders[:, :num_visible].unsqueeze(-1).expand(-1, -1, D))
        x = torch.cat([cls_token, visible], dim=1)

        output = self.encoder_mae.encoder(x)
        return self.encoder_mae.layernorm(output.last_hidden_state)

    @torch.compile
    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)

        x_buffer = x[:, :self.buffer_size, :]
        x_visible = x[:, self.buffer_size:, :]

        num_masked = ids_restore.shape[1] - x_visible.shape[1]
        mask_tokens = self.mask_token.repeat(x.shape[0], num_masked, 1).to(x.dtype)

        x_full = torch.cat([x_visible, mask_tokens], dim=1)
        x_full = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x_full.shape[2]))
        x = torch.cat([x_buffer, x_full], dim=1)

        x = x + self.decoder_pos_emb

        for block in self.decoder_block:
            x = block(x)

        decoded = self.decoder_norm(x)
        decoded = decoded[:, self.buffer_size:]

        return decoded

    def forward(self, x, mask_orders=None, labels=None, num_visible=None, force_unconditional=False):
        self.encoder_mae.eval()
        B, C, H, W = x.shape

        # [-1, 1] -> ImageNet normalized
        x_enc = (x + 1.0) / 2.0
        if H != self.img_size or W != self.img_size:
            x_enc = nn.functional.interpolate(x_enc, size=(self.img_size, self.img_size), mode='bicubic', align_corners=False)
        x_enc = (x_enc - self.encoder_mean) / self.encoder_std

        if self.training:
            # Training: built-in random masking, encoder only sees visible patches
            x, mask, ids_restore = self.forward_encoder_train(x_enc)
        else:
            # Generation: encode only visible patches selected by mask_orders
            x = self.encoder_generate(x_enc, mask_orders, num_visible)
            ids_restore = torch.argsort(mask_orders, dim=1)
            mask = torch.zeros(B, self.seq_len, device=x.device)
            mask.scatter_(1, mask_orders[:, num_visible:].long(), 1.0)

        # Class conditioning with label dropout
        if force_unconditional:
            class_embedding = self.fake_latent.expand(B, -1)
        else:
            class_embedding = self.class_emb(labels)
            if self.training:
                drop_mask = (torch.rand(B, device=x.device) < self.label_drop_prob).unsqueeze(-1).to(x.dtype)
                class_embedding = drop_mask * self.fake_latent + (1 - drop_mask) * class_embedding

        # Prepend buffer tokens with class conditioning, remove CLS token
        buffer_tokens = torch.zeros(B, self.buffer_size, x.shape[2], device=x.device, dtype=x.dtype)
        buffer_tokens[:] = class_embedding.unsqueeze(1)
        x = x[:, 1:, :]
        x = torch.cat([buffer_tokens, x], dim=1)

        z = self.forward_decoder(x, ids_restore)

        return z, mask

    @torch.no_grad()
    def update_ema(self):
        ema_decay = self.ema_decay
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params, source_params):
            targ.detach().mul_(ema_decay).add_(src, alpha=1 - ema_decay)
