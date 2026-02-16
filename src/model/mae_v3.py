import math
import numpy as np
from scipy import stats
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from torch.utils.checkpoint import checkpoint
from transformers import AutoImageProcessor, ViTMAEConfig, ViTMAEForPreTraining

from utils.utils import get_2d_sincos_pos_embed, patchify
    

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
        config.image_size = int(img_size)
        self.model_name = mae_config
        self.model_mae = ViTMAEForPreTraining.from_pretrained(self.model_name, config=config, ignore_mismatched_sizes=True)
        for param in self.model_mae.parameters():
            param.requires_grad = False
        self.model_mae.eval()

        self.ema_decay=ema_decay
        self.ema_params = None


    def initialize_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                # Xavier Uniform is standard for Transformers (often called Glorot)
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

        grid_size = int(self.seq_len ** 0.5)
        # pos_embed_grid = get_2d_sincos_pos_embed(self.encoder_dim, grid_size)
        # full_pos_embed = torch.zeros(self.seq_len + self.buffer_size, self.encoder_dim)
        # full_pos_embed[self.buffer_size:, :] = torch.from_numpy(pos_embed_grid).float()
        # self.encoder_pos_emb.data.copy_(full_pos_embed.unsqueeze(0))

        pos_embed_grid = get_2d_sincos_pos_embed(self.decoder_dim, grid_size)
        full_pos_embed = torch.zeros(self.seq_len + self.buffer_size, self.decoder_dim)
        full_pos_embed[self.buffer_size:, :] = torch.from_numpy(pos_embed_grid).float()
        self.decoder_pos_emb.data.copy_(full_pos_embed.unsqueeze(0))

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.class_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.fake_latent, std=.02)
    
    def forward_encoder(self, x, class_emb):
        mask_rate = stats.truncnorm((self.min_mask_rate - 1.0) / 0.25, 0, loc=1.0, scale=0.25).rvs(1)[0]
        self.model_mae.config.mask_ratio = mask_rate

        output = self.model_mae.vit(x, mask_rate=mask_rate, interpolate_pos_encoding=True)
        x = output.last_hidden_state
        mask = output.mask
        ids_restore = output.ids_restore

        return x, mask, ids_restore
    
    def encoder_generate(self, x, orders, num_visible, class_emb):
        self.model_mae.config.mask_ratio = 0

        x_embedding = self.model_mae.vit.embeddings(x)
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
        class_embedding = self.class_emb(labels)

        # Pretrain MAE need a range of 0 to 1
        x = (x + 1.0) / 2.0

        if H != self.img_size or W != self.img_size:
            x = nn.functional.interpolate(x, size=(self.img_size, self.img_size), mode='bicubic', align_corners=False)

        if self.training:
            x, mask, ids_restore = self.forward_encoder(x, class_embedding)
        else:
            x = self.encoder_generate(x, mask_orders, num_visible, class_embedding)
            mask = torch.zeros(B, self.seq_len, device=x.device)
            ids_restore = torch.argsort(mask_orders, dim=1)
            indices_to_mask = mask_orders[:, num_visible:]
            mask.scatter_(1, indices_to_mask.long(), 1.0)

        z, z_pixels = self.forward_decoder(x, ids_restore)

        return z, mask, z_pixels
    
    def random_masking(self, x, orders, min_mask_rate=0.7):
        bsz, seq_len, embed_dim = x.shape
        mask_rate = min_mask_rate
        mask_rate = stats.truncnorm((min_mask_rate - 1.0) / 0.25, 0, loc=1.0, scale=0.25).rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                                src=torch.ones(bsz, seq_len, device=x.device))
        return mask

    
    @torch.no_grad()
    def update_ema(self):
        ema_decay = self.ema_decay
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params, source_params):
            targ.detach().mul_(ema_decay).add_(src, alpha=1 - ema_decay)