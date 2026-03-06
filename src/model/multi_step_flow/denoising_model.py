import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from model.model_utils import Attention, CrossAttention, RMSNorm, SwiGLUFFN, TimestepEmbedder

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class ResBlock(nn.Module):
    """
    A residual block with adaLN.
    """
    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h
    
class FinalLayer(nn.Module):
    """
    The final layer.
    """
    def __init__(self, hidden_dim, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, patch_size * patch_size * out_channels, bias=True)
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
    
class AttenBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x,  c, feat_rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa.unsqueeze(1), scale_msa.unsqueeze(1)), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1)))
        return x
    
class InContextBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        mlp_ratio=4.0,
        attn_drop=0.1,
        proj_drop=0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.attn_ln = RMSNorm(hidden_dim, eps=1e-6)
        self.attn = Attention(hidden_dim, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.mlp_ln = RMSNorm(hidden_dim, eps=1e-6)
        self.mlp = SwiGLUFFN(hidden_dim, int(hidden_dim * mlp_ratio), drop=proj_drop)

    @torch.compile
    def forward(self, x, feat_rope=None):
        x = x + self.attn(self.attn_ln(x), rope=feat_rope)
        x = x + self.mlp(self.mlp_ln(x))
        return x

class DenoisingModel(nn.Module):
    """
    The diffusion model
    """
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        channels=3,
        num_classes=1000,
        hidden_dim=1024,
        depth=6,
        dropout=0.0,
        z_hidden_dim=768,
        denoiser_type = 'ada_ln',
        bottleneck_dim=None
    ):
        super().__init__()
        self.in_channels = channels
        self.out_channels = channels
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_heads = hidden_dim // 64
        self.img_size = img_size
        self.denoiser_type = denoiser_type
        self.bottleneck_dim = bottleneck_dim

        self.embedding_dim = channels * patch_size**2
        self.num_patches = (img_size // patch_size) ** 2

        if bottleneck_dim is not None:
            self.x_proj = nn.Sequential(
                nn.Linear(self.embedding_dim, bottleneck_dim, bias=False),
                nn.Linear(bottleneck_dim, hidden_dim, bias=True),
            )
        else:
            self.x_proj = nn.Linear(self.embedding_dim, hidden_dim)
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.z_proj = nn.Linear(z_hidden_dim, hidden_dim)

        if denoiser_type == 'ada_ln':
            self.blocks = nn.ModuleList([
                ResBlock(hidden_dim, depth_index=i, total_depth=depth)
                for i in range(depth)
            ])
            self.final_layer = FinalLayer(hidden_dim, patch_size, channels)
            self.initialize_weights_ada_ln()
        elif denoiser_type == 'in_context':
            self.blocks = nn.ModuleList([
                InContextBlock(hidden_dim, hidden_dim // 64)
                for i in range(depth)
            ])
            self.final_layer = nn.Linear(hidden_dim, self.embedding_dim, bias=True)
            self.initialize_weights_in_context()
        elif denoiser_type == 'attn':
            self.blocks = nn.ModuleList([
                AttenBlock(hidden_dim, hidden_dim // 64)
                for i in range(depth)
            ])
            self.final_layer = FinalLayer(hidden_dim, patch_size, channels)
            self.initialize__attn_weights()
        elif denoiser_type == 'add':
            self.blocks = nn.ModuleList([
                AttenBlock(hidden_dim, hidden_dim // 64)
                for i in range(depth)
            ])
            self.final_layer = FinalLayer(hidden_dim, patch_size, channels)
            self.initialize__attn_weights()
        
    def initialize__attn_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Specific normal initialization for projections and embedders
        if self.bottleneck_dim is not None:
            nn.init.xavier_uniform_(self.x_proj[0].weight)
            nn.init.xavier_uniform_(self.x_proj[1].weight)
            nn.init.constant_(self.x_proj[1].bias, 0)
        else:
            nn.init.normal_(self.x_proj.weight, std=0.02)
        nn.init.normal_(self.z_proj.weight, std=0.02)

        # Timestep MLP initialization (standard in diffusion models)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def initialize_weights_in_context(self):
        # Basic Xavier initialization for all Linear layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Specific normal initialization for projections and embedders
        if self.bottleneck_dim is not None:
            nn.init.xavier_uniform_(self.x_proj[0].weight)
            nn.init.xavier_uniform_(self.x_proj[1].weight)
            nn.init.constant_(self.x_proj[1].bias, 0)
        else:
            nn.init.normal_(self.x_proj.weight, std=0.02)
        nn.init.normal_(self.z_proj.weight, std=0.02)

        # Timestep MLP initialization (standard in diffusion models)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            # For Attention: Zero the output projection
            nn.init.constant_(block.attn.proj.weight, 0)
            if block.attn.proj.bias is not None:
                nn.init.constant_(block.attn.proj.bias, 0)
            
            # Zero the output layer of the SwiGLU MLP
            nn.init.constant_(block.mlp.w3.weight, 0)
            if block.mlp.w3.bias is not None:
                nn.init.constant_(block.mlp.w3.bias, 0)

        nn.init.constant_(self.final_layer.weight, 0)
        nn.init.constant_(self.final_layer.bias, 0)

    def initialize_weights_ada_ln(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            # Zero adaLN modulation weights, then set non-zero gate bias
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            
            # MLP (w3) keeps Xavier init — gate at 1/depth controls magnitude.
            block.initialize_gate_bias()

            # Zero x0_proj so input injection starts inactive
            nn.init.constant_(block.x0_proj.weight, 0)
            nn.init.constant_(block.x0_proj.bias, 0)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward_in_context(self, x, z, t):
        """
        x: (B*N, D)
        z: (B*N, D')
        y: (B*N, Cls_num)
        t: (B*N, 1)
        """
        x = self.x_proj(x)  
        t = self.t_embedder(t)      
        z = self.z_proj(z)

        c = t + z

        c = c.unsqueeze(1)
        x = x.unsqueeze(1)
        x = torch.cat([x, c], dim=1)

        for block in self.blocks:
            x = block(x)

        x = x[:, 0, :]
        x = self.final_layer(x)

        return x
    
    def forward_attn(self, x, z, t):
        """
        x: (B*N, D)
        z: (B*N, D')
        y: (B*N, Cls_num)
        t: (B*N, 1)
        """
        x = self.x_proj(x)  
        t = self.t_embedder(t)      
        z = self.z_proj(z)

        c = t + z

        x = x.unsqueeze(1)

        for block in self.blocks:
            x = block(x, c)

        x = x[:, 0, :]
        x = self.final_layer(x, c)

        return x
    
    def forward_ada_ln(self, x, z, t):
        """
        x: (B*N, D)
        z: (B*N, D')
        t: (B*N, 1)
        """

        x = self.x_proj(x)
        t = self.t_embedder(t)
        z = self.z_proj(z)
        c = t + z

        for block in self.blocks:
            x = block(x, c)

        x = self.final_layer(x, c)
        return x

    def forward_add(self, x, z, t):
        """
        x: (B*N, D)
        z: (B*N, D')
        t: (B*N, 1)
        """
        x = self.x_proj(x)
        t = self.t_embedder(t)
        z = self.z_proj(z)

        c = t  # adaLN condition is timestep only
        x = x.unsqueeze(1)
        z = z.unsqueeze(1)

        for block in self.blocks:
            x = block(x + z, c)

        x = x[:, 0, :]
        x = self.final_layer(x, c)

        return x

    def forward(self, x, z, t):
        if self.denoiser_type == 'ada_ln':
            x = self.forward_ada_ln(x, z, t)
        elif self.denoiser_type == 'in_context':
            x = self.forward_in_context(x, z, t)
        elif self.denoiser_type == 'attn':
            x = self.forward_attn(x, z, t)
        elif self.denoiser_type == 'add':
            x = self.forward_add(x, z, t)
        else:
            raise NotImplementedError
        return x



