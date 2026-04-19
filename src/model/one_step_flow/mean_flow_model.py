import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model_utils import RMSNorm, SwiGLUFFN, TimestepEmbedder


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class ResBlock(nn.Module):
    """adaLN ResBlock for MeanFlow."""
    def __init__(self, hidden_dim, hidden_ratio=4.0, proj_drop=0.0):
        super().__init__()
        self.norm = RMSNorm(hidden_dim, eps=1e-6)
        self.mlp = SwiGLUFFN(hidden_dim, hidden_ratio * hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim, bias=True),
        )

    def forward(self, x, c):
        shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=-1)
        return x + gate * self.mlp(modulate(self.norm(x), shift, scale))


class FinalLayer(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.norm_final = RMSNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class MeanFlowModel(nn.Module):
    """
    MeanFlow denoising network with z fusion, adaLN-conditioning and dual output heads:
        u : predicted average velocity (mean flow)
        v : predicted instantaneous velocity (auxiliary head).
    """
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        channels=3,
        hidden_dim=1024,
        depth=6,
        aux_head_depth=2,
        hidden_ratio=4.0,
        dropout=0.0,
        z_hidden_dim=768,
        bottleneck_dim=None,
        latent_dim=None,
        pred_type='x',
        t_eps=0.05,
    ):
        super().__init__()
        self.in_channels = channels
        self.out_channels = channels
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.img_size = img_size
        self.pred_type = pred_type
        self.t_eps = t_eps

        self.embedding_dim = latent_dim if latent_dim is not None else channels * patch_size ** 2
        self.num_patches = (img_size // patch_size) ** 2

        # JiT-style bottleneck on the channel dimention.
        self.bottleneck_dim = bottleneck_dim
        if self.bottleneck_dim is not None:
            self.x_proj = nn.Sequential(
                nn.Linear(self.embedding_dim, self.bottleneck_dim, bias=False),
                nn.Linear(self.bottleneck_dim, hidden_dim, bias=True),
            )
        else:
            self.x_proj = nn.Linear(self.embedding_dim, hidden_dim)

        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.h_embedder = TimestepEmbedder(hidden_dim)
        self.z_proj = nn.Linear(z_hidden_dim, hidden_dim)
        self.fusion_emb = nn.Linear(2 * hidden_dim, hidden_dim)

        shared_depth = max(0, depth - aux_head_depth)
        self.shared_blocks = nn.ModuleList([
            ResBlock(hidden_dim, hidden_ratio=hidden_ratio, proj_drop=dropout) for _ in range(shared_depth)
        ])
        self.u_blocks = nn.ModuleList([
            ResBlock(hidden_dim, hidden_ratio=hidden_ratio, proj_drop=dropout) for _ in range(aux_head_depth)
        ])
        self.v_blocks = nn.ModuleList([
            ResBlock(hidden_dim, hidden_ratio=hidden_ratio, proj_drop=dropout) for _ in range(aux_head_depth)
        ])

        self.u_final_layer = FinalLayer(hidden_dim, self.embedding_dim)
        self.v_final_layer = FinalLayer(hidden_dim, self.embedding_dim)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic_init)

        if isinstance(self.x_proj, nn.Sequential):
            nn.init.xavier_uniform_(self.x_proj[0].weight)
            nn.init.xavier_uniform_(self.x_proj[1].weight)
            nn.init.constant_(self.x_proj[1].bias, 0)
        else:
            nn.init.normal_(self.x_proj.weight, std=0.02)
        nn.init.normal_(self.z_proj.weight, std=0.02)
        nn.init.normal_(self.fusion_emb.weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.h_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.h_embedder.mlp[2].weight, std=0.02)

        for block in list(self.shared_blocks) + list(self.u_blocks) + list(self.v_blocks):
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        for final in (self.u_final_layer, self.v_final_layer):
            nn.init.constant_(final.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(final.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(final.linear.weight, 0)
            nn.init.constant_(final.linear.bias, 0)

    def forward(self, x, z, t, h):
        """
        Args:
            x : (BN, D)  noisy sample at time t (per-token)
            z : (BN, D') AR conditioning (per-token)
            t : (BN,)    current time
            h : (BN,)    time gap, h = t - r (>= 0)
        Returns:
            u : (BN, D)  predicted average velocity over [t-h, t]
            v : (BN, D)  predicted instantaneous velocity at time t (aux)
        """
        x_emb = self.x_proj(x)
        z_emb = self.z_proj(z)
        t_emb = self.t_embedder(t)
        h_emb = self.h_embedder(h)

        c = t_emb + h_emb + z_emb
        x_in = self.fusion_emb(torch.cat((x_emb, z_emb), dim=-1))

        for block in self.shared_blocks:
            x_in = block(x_in, c)

        u_h = x_in
        v_h = x_in
        for block in self.u_blocks:
            u_h = block(u_h, c)
        for block in self.v_blocks:
            v_h = block(v_h, c)

        u_raw = self.u_final_layer(u_h, c)
        v_raw = self.v_final_layer(v_h, c)

        if self.pred_type == 'x':
            # x-prediction (matches pMF)
            t_div = t.clamp(min=self.t_eps, max=1.0).view(-1, 1)
            u = (x - u_raw) / t_div
            v = (x - v_raw) / t_div
        elif self.pred_type == 'v':
            u = u_raw
            v = v_raw
        else:
            raise ValueError(f"Unknown pred_type: '{self.pred_type}'.")

        return u, v
