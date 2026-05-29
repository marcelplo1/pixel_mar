import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model_utils import RMSNorm, SwiGLUFFN, TimestepEmbedder

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class ResBlock(nn.Module):
    def __init__(self, hidden_dim, hidden_ratio=4.0, proj_drop=0.0):
        super().__init__()
        self.norm = RMSNorm(hidden_dim, eps=1e-6)
        mlp_hidden_dim = int(hidden_dim * hidden_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, hidden_dim),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim, bias=True)
        )

    @torch.compile
    def forward(self, x, c):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(3, dim=-1)
        x = x + gate_mlp * self.mlp(modulate(self.norm(x), shift_mlp, scale_mlp))
        return x



class FinalLayer(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.norm_final = RMSNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )

    @torch.compile
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DenoisingModel(nn.Module):
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        channels=3,
        num_classes=1000,
        hidden_dim=1024,
        depth=6,
        hidden_ratio=4.0,
        dropout=0.0,
        z_hidden_dim=768,
        denoiser_type='ada_ln',
        bottleneck_dim=None,
        latent_dim=None
    ):
        super().__init__()
        self.in_channels = channels
        self.out_channels = channels
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_heads = hidden_dim // 64
        self.img_size = img_size
        self.denoiser_type = denoiser_type

        self.embedding_dim = latent_dim if latent_dim is not None else channels * patch_size**2
        self.num_patches = (img_size // patch_size) ** 2

        self.bottleneck_dim = bottleneck_dim
        if self.bottleneck_dim is not None:
            self.x_proj = nn.Sequential(
                nn.Linear(self.embedding_dim, self.bottleneck_dim, bias=False),
                nn.Linear(self.bottleneck_dim, hidden_dim, bias=True),
            )
        else:
            self.x_proj = nn.Linear(self.embedding_dim, hidden_dim)

        self.t_embedder = TimestepEmbedder(hidden_dim)

        self.z_proj = nn.Linear(z_hidden_dim, hidden_dim)
        if denoiser_type == 'ada_ln_fusion' or denoiser_type == 'ada_ln_fusion_only':
            self.fusion_emb = nn.Linear(2 * hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResBlock(hidden_dim, hidden_ratio=hidden_ratio, proj_drop=dropout)
            for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_dim, self.embedding_dim)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        if isinstance(self.x_proj, nn.Sequential):
            nn.init.xavier_uniform_(self.x_proj[0].weight)
            nn.init.xavier_uniform_(self.x_proj[1].weight)
            nn.init.constant_(self.x_proj[1].bias, 0)
        else:
            nn.init.normal_(self.x_proj.weight, std=0.02)

        nn.init.normal_(self.z_proj.weight, std=0.02)
        if self.denoiser_type == 'ada_ln_fusion' or self.denoiser_type == 'ada_ln_fusion_only':
            nn.init.normal_(self.fusion_emb.weight, std=0.02)
            nn.init.zeros_(self.fusion_emb.weight[:, self.hidden_dim:])  # zero the z half
            nn.init.zeros_(self.fusion_emb.bias)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, z, t):
        return self._forward_ada_ln(x, z, t)

    def _forward_ada_ln(self, x, z, t):
        """
        x: (B*N, D)
        z: (B*N, z_dim)
        t: (B*N,)
        """
        x = self.x_proj(x)
        t = self.t_embedder(t)
        z = self.z_proj(z)

        if self.denoiser_type == 'ada_ln_fusion_only':
            x = self.fusion_emb(torch.cat((x, z), dim=-1))
            c = t
        elif self.denoiser_type == 'ada_ln_fusion':
            x = self.fusion_emb(torch.cat((x, z), dim=-1))
            c = t + z
        else:
            c = t + z

        for block in self.blocks:
            x = block(x, c)

        x = self.final_layer(x, c)
        return x
