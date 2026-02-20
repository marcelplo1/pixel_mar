import math
import numpy as np
from scipy import stats
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoImageProcessor, ViTMAEConfig, ViTMAEForPreTraining
    

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
            grad_ckpt = False,
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
        self.grad_ckpt = grad_ckpt

        # self.encoder_proj = nn.Sequential(
        #     nn.Linear(self.embed_dim, bottleneck_dim),
        #     nn.SiLU(),
        #     nn.Linear(bottleneck_dim, hidden_dim)
        # )

        self.class_emb = nn.Embedding(num_classes, encoder_dim)

        proc = AutoImageProcessor.from_pretrained(mae_config)
        self.encoder_mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1)
        self.encoder_std = torch.tensor(proc.image_std).view(1, 3, 1, 1)

        config = ViTMAEConfig.from_pretrained(mae_config)
        config.patch_size = int(patch_size)
        config.num_channels = int(channels)
        self.model_name = mae_config
        self.model_mae = ViTMAEForPreTraining.from_pretrained(self.model_name, config=config)

        # Interpolate decoder pos_embed from pretrained resolution to target resolution
        pretrain_num_patches = (config.image_size // patch_size) ** 2
        if self.seq_len != pretrain_num_patches:
            old_pos = self.model_mae.decoder.decoder_pos_embed.data  # [1, 1+old_patches, D]
            cls_pos = old_pos[:, :1, :]
            patch_pos = old_pos[:, 1:, :]
            grid_old = int(pretrain_num_patches ** 0.5)
            grid_new = img_size // patch_size
            D = patch_pos.shape[-1]
            patch_pos = patch_pos.reshape(1, grid_old, grid_old, D).permute(0, 3, 1, 2)
            patch_pos = nn.functional.interpolate(
                patch_pos.float(), size=(grid_new, grid_new), mode='bicubic', align_corners=False
            )
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, self.seq_len, D)
            self.model_mae.decoder.decoder_pos_embed = nn.Parameter(
                torch.cat([cls_pos, patch_pos], dim=1), requires_grad=False
            )

        for param in self.model_mae.parameters():
            param.requires_grad = False
        self.model_mae.eval()

        self.ema_decay=ema_decay
        self.ema_params = None
    
    def forward_encoder(self, x):
        mask_rate = stats.truncnorm((self.min_mask_rate - 1.0) / 0.25, 0, loc=1.0, scale=0.25).rvs(1)[0]
        self.model_mae.config.mask_ratio = mask_rate

        output = self.model_mae.vit(x, mask_rate=mask_rate, interpolate_pos_encoding=True)
        x = output.last_hidden_state
        mask = output.mask
        ids_restore = output.ids_restore

        return x, mask, ids_restore
    
    def encoder_generate(self, x, orders, num_visible):
        self.model_mae.config.mask_ratio = 0
        noise = torch.arange(self.seq_len).unsqueeze(0).expand(x.shape[0], -1).to(x.device).float()

        x_embedding = self.model_mae.vit.embeddings(x, noise=noise, interpolate_pos_encoding=True)
        x_embedding = x_embedding[0] if isinstance(x_embedding, tuple) else x_embedding

        B, N, D = x_embedding.shape
        cls_token = x_embedding[:, :1, :]
        img_tokens = x_embedding[:, 1:, :]

        x = torch.gather(img_tokens, dim=1, index=orders[:, :num_visible].unsqueeze(-1).expand(-1, -1, D))
        x = torch.cat([cls_token, x], dim=1)

        output = self.model_mae.vit.encoder(x)
        return output.last_hidden_state
    
    def forward_decoder(self, x, ids_restore):    
        x_dec = self.model_mae.decoder.decoder_embed(x)
        
        mask_tokens = self.model_mae.decoder.mask_token.repeat(
            x_dec.shape[0], ids_restore.shape[1] - x_dec.shape[1] + 1, 1
        )
        
        x_full = torch.cat([x_dec[:, 1:, :], mask_tokens], dim=1) 
        x_full = torch.gather(
            x_full, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_dec.shape[2])
        )
        
        x_dec = torch.cat([x_dec[:, :1, :], x_full], dim=1)
        x_dec = x_dec + self.model_mae.decoder.decoder_pos_embed

        for layer in self.model_mae.decoder.decoder_layers:
            x_dec = layer(x_dec)
            
        x_dec = self.model_mae.decoder.decoder_norm(x_dec)
        logits = self.model_mae.decoder.decoder_pred(x_dec[:, 1:, :])

        return x_dec[:, 1:], logits

    def forward(self, x, mask_orders, labels, num_visible=None):
        self.model_mae.eval()
        B, C, H, W = x.shape

        # Pretrain MAE need a range of 0 to 1
        x = (x + 1.0) / 2.0

        if H != self.img_size or W != self.img_size:
            x = nn.functional.interpolate(x, size=(self.img_size, self.img_size), mode='bicubic', align_corners=False)

        if self.training:
            x, mask, ids_restore = self.forward_encoder(x)
        else:
            x = self.encoder_generate(x, mask_orders, num_visible)
            mask = torch.zeros(B, self.seq_len, device=x.device)
            ids_restore = torch.argsort(mask_orders, dim=1)
            indices_to_mask = mask_orders[:, num_visible:]
            mask.scatter_(1, indices_to_mask.long(), 1.0)

        z, z_pixels = self.forward_decoder(x, ids_restore)

        return z, mask, z_pixels

    @torch.no_grad()
    def update_ema(self):
        ema_decay = self.ema_decay
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params, source_params):
            targ.detach().mul_(ema_decay).add_(src, alpha=1 - ema_decay)