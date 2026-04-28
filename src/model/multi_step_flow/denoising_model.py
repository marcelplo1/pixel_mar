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


class CrossAttnResBlock(nn.Module):
    def __init__(self, hidden_dim, z_hidden_dim, num_heads=8, hidden_ratio=4.0, proj_drop=0.0):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Cross-attention: x (query) attends to z_seq (key/value)
        self.norm_x = RMSNorm(hidden_dim)
        self.norm_z = RMSNorm(z_hidden_dim)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(z_hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(z_hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # adaLN MLP conditioned on timestep only
        self.norm_mlp = RMSNorm(hidden_dim)
        mlp_hidden_dim = int(hidden_dim * hidden_ratio)
        self.mlp = SwiGLUFFN(hidden_dim, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim, bias=True)
        )

    def compute_kv(self, z_seq):
        """Precompute K and V from a fixed z_seq — call once before the ODE loop."""
        B, N, _ = z_seq.shape
        k = self.k_proj(self.norm_z(z_seq))
        v = self.v_proj(z_seq)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_norm(k)
        return k, v

    def forward(self, x, z_seq, c, kv=None):
        # x:     (B, L, H)      — noisy tokens (L=N train, L=1 inference)
        # z_seq: (B, N, z_dim)  — full AR context sequence (unused when kv is provided)
        # c:     (B, L, H)      — timestep conditioning
        # kv:    optional precomputed (k, v) from compute_kv()
        B, L, H = x.shape

        q = self.q_proj(self.norm_x(x))
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)

        if kv is not None:
            k, v = kv
        else:
            k = self.k_proj(self.norm_z(z_seq))
            v = self.v_proj(z_seq)
            k = k.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_norm(k)

        x_attn = F.scaled_dot_product_attention(q, k, v)
        x_attn = x_attn.transpose(1, 2).reshape(B, L, H)
        x = x + self.out_proj(x_attn)

        shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=-1)
        x = x + gate * self.mlp(modulate(self.norm_mlp(x), shift, scale))
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

        if denoiser_type == 'cross_attn':
            self.query_pos_emb = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))
            self.blocks = nn.ModuleList([
                CrossAttnResBlock(hidden_dim, z_hidden_dim, num_heads=self.num_heads,
                                  hidden_ratio=hidden_ratio, proj_drop=dropout)
                for _ in range(depth)
            ])
        else:
            self.z_proj = nn.Linear(z_hidden_dim, hidden_dim)
            if denoiser_type == 'ada_ln_fusion':
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

        if self.denoiser_type == 'cross_attn':
            nn.init.normal_(self.query_pos_emb, std=0.02)
            for block in self.blocks:
                # Zero-init out_proj so cross-attn residual starts at zero
                nn.init.zeros_(block.out_proj.weight)
                nn.init.zeros_(block.out_proj.bias)
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        else:
            nn.init.normal_(self.z_proj.weight, std=0.02)
            if self.denoiser_type == 'ada_ln_fusion':                                                                                                                                               
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

    def precompute_kv(self, z_seq):
        """Precompute K/V for all blocks from a fixed z_seq — call once before the ODE loop."""
        return [block.compute_kv(z_seq) for block in self.blocks]

    def forward(self, x, z, t, kv_cache=None, token_positions=None):
        if self.denoiser_type == 'cross_attn':
            return self._forward_cross_attn(x, z, t, kv_cache=kv_cache, token_positions=token_positions)
        else:
            return self._forward_ada_ln(x, z, t)

    def _forward_cross_attn(self, x, z_seq, t, kv_cache=None, token_positions=None):
        """
        x:               (B * L, D)    — L=N during training, L=1 during inference
        z_seq:           (B, N, z_dim) — full AR context sequence per image
        t:               (B * L,)
        kv_cache:        optional list of (k, v) per block from precompute_kv()
        token_positions: (B,) - sequence position of each token during sampling
        """
        B_eff = z_seq.shape[0]
        L = x.shape[0] // B_eff

        x = self.x_proj(x)          # (B_eff * L, H)
        t_emb = self.t_embedder(t)  # (B_eff * L, H)

        x = x.reshape(B_eff, L, -1)
        c = t_emb.reshape(B_eff, L, -1)

        if token_positions is not None:
            # Inference: L=1 token
            x = x + self.query_pos_emb[:, token_positions, :].transpose(0, 1)  # (B_eff, 1, H)
        else:
            # Training: L=N tokens in canonical order 0..N-1
            x = x + self.query_pos_emb[:, :L, :]

        for i, block in enumerate(self.blocks):
            kv = kv_cache[i] if kv_cache is not None else None
            x = block(x, z_seq, c, kv=kv)

        x = x.reshape(B_eff * L, -1)
        c = c.reshape(B_eff * L, -1)
        return self.final_layer(x, c)

    def _forward_ada_ln(self, x, z, t):
        """
        x: (B*N, D)
        z: (B*N, z_dim)
        t: (B*N,)
        """
        x = self.x_proj(x)
        t = self.t_embedder(t)
        z = self.z_proj(z)
        c = t + z

        if self.denoiser_type == 'ada_ln_fusion':
            x = self.fusion_emb(torch.cat((x, z), dim=-1))

        for block in self.blocks:
            x = block(x, c)

        x = self.final_layer(x, c)
        return x
