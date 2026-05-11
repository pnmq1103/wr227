"""
MARL-Gated RALA ViT Backbone for MMDetection.

Flat 16-layer ViT with projection layers that create a 4-level feature
pyramid for FPN-based detection heads (RetinaNet, Mask R-CNN).

Architecture:
    Input → PatchEmbed → 16× EncoderBlock (shared router) → Norm
    
    Features extracted at layers [3, 7, 11, 15] (every 4 layers).
    Each is reshaped to 2D and projected to produce 4 FPN levels:
        Level 0: TransposeConv 2× upsample  (stride = patch_size / 2)
        Level 1: Identity + 1×1 conv         (stride = patch_size)
        Level 2: Conv stride 2 downsample     (stride = patch_size × 2)
        Level 3: Conv stride 2 × 2 downsample (stride = patch_size × 4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

try:
    from mmengine.model import BaseModule
    from mmdet.registry import MODELS
    HAS_MMDET = True
except ImportError:
    # Fallback for standalone testing without MMDetection
    BaseModule = nn.Module
    HAS_MMDET = False
    warnings.warn("MMDetection not found. Backbone will not be registered.")

from .attention import (
    SharedActorCritic,
    EncoderBlock,
    PatchEmbedding,
)


class FPNProjection(nn.Module):
    """
    Projects flat ViT features at a single layer into one FPN level.
    Reshapes tokens to 2D, then applies scale-specific convolutions.
    """

    def __init__(self, in_channels, out_channels, scale_factor):
        """
        Args:
            in_channels: ViT d_model (e.g. 256)
            out_channels: FPN level channels (e.g. 256)
            scale_factor: spatial scale relative to patch grid
                2.0 → upsample 2× (for stride = patch_size/2)
                1.0 → same resolution (for stride = patch_size)
                0.5 → downsample 2× (for stride = patch_size*2)
                0.25 → downsample 4× (for stride = patch_size*4)
        """
        super().__init__()
        self.scale_factor = scale_factor

        layers = []
        if scale_factor == 2.0:
            # Upsample 2×
            layers.append(nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.GELU())
        elif scale_factor == 1.0:
            # Same resolution
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.GELU())
        elif scale_factor == 0.5:
            # Downsample 2×
            layers.append(nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.GELU())
        elif scale_factor == 0.25:
            # Downsample 4× (two stride-2 convs)
            layers.append(nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.GELU())
            layers.append(nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.GELU())
        else:
            raise ValueError(f"Unsupported scale_factor: {scale_factor}")

        self.proj = nn.Sequential(*layers)

    def forward(self, tokens, H, W):
        """
        Args:
            tokens: (B, N, C) where N = H * W
            H, W: spatial dimensions of the patch grid
        Returns:
            feature_map: (B, out_channels, H', W')
        """
        B, N, C = tokens.shape
        x = tokens.transpose(1, 2).view(B, C, H, W)
        return self.proj(x)


def _build_backbone(base_cls):
    """Factory to build the backbone class with the appropriate base."""

    class MARLRALABackbone(base_cls):
        """
        Flat 16-layer MARL-Gated RALA ViT with FPN projection layers.

        Extracts intermediate features at 4 equally-spaced depths and
        projects them into a 4-level spatial feature pyramid for FPN.

        The router can be frozen (w=1.0) during early training epochs.
        """

        def __init__(self,
                     patch_size=16,
                     in_chans=3,
                     d_model=256,
                     depth=16,
                     num_heads=8,
                     chunk_size=16,
                     drop_path_rate=0.1,
                     out_channels=(256, 256, 256, 256),
                     fpn_scales=(2.0, 1.0, 0.5, 0.25),
                     out_indices=(3, 7, 11, 15),
                     freeze_router=False,
                     init_cfg=None):
            """
            Args:
                patch_size: Patch embedding stride
                in_chans: Input image channels
                d_model: Transformer embedding dimension
                depth: Number of transformer layers
                num_heads: Number of attention heads
                chunk_size: Chunk size for chunkwise attention
                drop_path_rate: Maximum stochastic depth rate
                out_channels: Channel dims for each FPN level
                fpn_scales: Spatial scale for each FPN projection
                out_indices: Layer indices (0-based) to extract features from
                freeze_router: If True, hardcode w=1.0 (no MARL routing)
                init_cfg: MMEngine init config for weight loading
            """
            if HAS_MMDET:
                super().__init__(init_cfg=init_cfg)
            else:
                super().__init__()

            self.d_model = d_model
            self.depth = depth
            self.out_indices = out_indices
            self.freeze_router = freeze_router
            self.patch_size = patch_size

            # --- Patch Embedding ---
            self.patch_embed = PatchEmbedding(
                patch_size=patch_size, in_chans=in_chans, embed_dim=d_model)

            # --- Learnable positional embedding (interpolatable) ---
            # Initialize for a default size; will be interpolated at runtime
            self._default_H = 14  # 224 / 16
            self._default_W = 14
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self._default_H * self._default_W, d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

            # --- Encoder Blocks ---
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
            self.blocks = nn.ModuleList([
                EncoderBlock(d_model, num_heads, chunk_size, drop_path=dpr[i])
                for i in range(depth)
            ])
            self.norm = nn.LayerNorm(d_model)

            # --- Shared Actor-Critic Router ---
            self.router = SharedActorCritic(d_model)

            # --- FPN Projection Layers ---
            assert len(out_indices) == len(out_channels) == len(fpn_scales), \
                "out_indices, out_channels, fpn_scales must have same length"
            self.fpn_projections = nn.ModuleList([
                FPNProjection(d_model, oc, sf)
                for oc, sf in zip(out_channels, fpn_scales)
            ])

            # --- Layer norms for each extracted feature ---
            self.fpn_norms = nn.ModuleList([
                nn.LayerNorm(d_model) for _ in out_indices
            ])

        def _interpolate_pos_embed(self, H, W):
            """Interpolate positional embeddings to match current spatial size."""
            N = H * W
            if N == self.pos_embed.shape[1]:
                return self.pos_embed

            pos = self.pos_embed.reshape(
                1, self._default_H, self._default_W, self.d_model
            ).permute(0, 3, 1, 2)  # (1, C, H0, W0)

            pos = F.interpolate(
                pos, size=(H, W), mode='bicubic', align_corners=False
            )
            pos = pos.permute(0, 2, 3, 1).reshape(1, N, self.d_model)
            return pos

        def forward(self, x):
            """
            Args:
                x: (B, 3, H_img, W_img) input images

            Returns:
                tuple of 4 feature maps: [(B, C_i, H_i, W_i), ...]
                    for FPN consumption

            Side effects:
                Stores routing metadata in self._routing_info for PPO hook
            """
            B = x.shape[0]

            # --- Patch embedding ---
            tokens, H, W = self.patch_embed(x)  # (B, H*W, d_model)

            # --- Add (interpolated) positional embedding ---
            pos = self._interpolate_pos_embed(H, W)
            tokens = tokens + pos

            # --- Store routing info for PPO ---
            w_list = []
            log_prob_list = []
            value_list = []
            mu_list = []
            sigma_list = []

            # --- Run through all blocks, extract at out_indices ---
            fpn_features = {}
            deterministic = not self.training or self.freeze_router

            for i, block in enumerate(self.blocks):
                # Router: get gating weights
                w, log_prob, value, mu, sigma = self.router(
                    tokens, deterministic=deterministic
                )

                if self.freeze_router:
                    w_gating = torch.ones_like(w)
                else:
                    w_gating = w

                tokens, _ = block(tokens, w_gating, H, W,
                                  use_dilution=(not self.freeze_router))

                w_list.append(w)
                log_prob_list.append(log_prob)
                value_list.append(value)
                mu_list.append(mu)
                sigma_list.append(sigma)

                if i in self.out_indices:
                    fpn_features[i] = tokens

            # --- Final norm on last layer output ---
            tokens = self.norm(tokens)
            if self.out_indices[-1] == self.depth - 1:
                fpn_features[self.out_indices[-1]] = tokens

            # --- Project to FPN levels ---
            outputs = []
            for idx, layer_idx in enumerate(self.out_indices):
                feat = self.fpn_norms[idx](fpn_features[layer_idx])
                feat_2d = self.fpn_projections[idx](feat, H, W)
                outputs.append(feat_2d)

            # --- Store routing metadata for PPO hook ---
            has_lp = log_prob_list[0] is not None
            self._routing_info = {
                'w_t': torch.stack(w_list, dim=1),           # (B, L, N)
                'log_probs': (torch.stack(log_prob_list, dim=1)
                              if has_lp else None),          # (B, L, N)
                'values': torch.stack(value_list, dim=1),    # (B, L, N)
                'mu': torch.stack(mu_list, dim=1),           # (B, L, N)
                'sigma': torch.stack(sigma_list, dim=1),     # (B, L, N)
            }

            return tuple(outputs)

    return MARLRALABackbone


# Build the class — registered if MMDetection is available
_BackboneClass = _build_backbone(BaseModule)

if HAS_MMDET:
    MODELS.register_module(name='MARLRALABackbone', module=_BackboneClass)

MARLRALABackbone = _BackboneClass
