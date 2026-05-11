"""
Core RALA attention modules for the MARL-gated Vision Transformer.

Extracted from marl_vit_v2.ipynb and adapted for:
- Variable spatial sizes (detection needs different resolutions)
- MMDetection integration
- Per-stage routing with independent Actor-Critic modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from einops import rearrange
import math


class SharedActorCritic(nn.Module):
    """
    Shared Actor-Critic for the swarm of N agents.

    - Each token is an independent agent; the AC processes them all in parallel.
    - The actor outputs a squashed Gaussian scalar weight w ∈ (0,1) per token.
    - The critic outputs a per-token value estimate.
    """

    def __init__(self, d_model: int):
        super().__init__()
        half = max(d_model // 2, 16)

        self.proj = nn.Linear(d_model, half)
        self.actor_mlp = nn.Sequential(nn.Linear(half, half), nn.ELU())
        self.mu_head = nn.Linear(half, 1)
        self.sigma_head = nn.Linear(half, 1)
        self.critic_mlp = nn.Sequential(
            nn.Linear(half, half), nn.ELU(), nn.Linear(half, 1)
        )

    def forward(self, local_features: torch.Tensor, deterministic: bool = False):
        """
        Args:
            local_features: (B, N, d_model)
            deterministic: if True, use mean (no sampling)
        Returns:
            w:        (B, N) gating weights in (0, 1)
            log_prob: (B, N) Jacobian-corrected log probability (None if deterministic)
            value:    (B, N) per-token value estimate
            mu:       (B, N) actor mean
            sigma:    (B, N) actor std
        """
        s = self.proj(local_features)
        h_actor = self.actor_mlp(s)
        mu = self.mu_head(h_actor).squeeze(-1)
        sigma = F.softplus(self.sigma_head(h_actor)).squeeze(-1) + 1e-5
        value = self.critic_mlp(s).squeeze(-1)

        if deterministic:
            z = mu
            log_prob = None
        else:
            dist = Normal(mu, sigma)
            z = dist.rsample()
            log_prob = dist.log_prob(z)

        w_raw = torch.tanh(z)
        w = (w_raw + 1.0) / 2.0

        if not deterministic:
            jacobian = torch.log(0.5 * (1.0 - w_raw.pow(2)) + 1e-5)
            log_prob = log_prob - jacobian

        return w, log_prob, value, mu, sigma


class ChunkwiseRALAAttention(nn.Module):
    """
    Chunkwise Linear Attention with MARL gating.

    Based on RALA but replaces α_j reweighting with MARL agent weights.
    Adapted from notebook to handle variable sequence lengths by auto-padding
    to the nearest multiple of chunk_size.
    """

    def __init__(self, d_model: int, head: int = 8, chunk_size: int = 16,
                 gamma: float = 0.1, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.head = head
        self.chunk_size = chunk_size
        self.gamma = gamma
        self.d_k = d_model // head

        # Fused Q/K/V/O projection (matches RALA's Conv2d(dim, dim*4, 1))
        self.qkvo = nn.Linear(d_model, d_model * 4)
        # LePE: 5×5 depthwise conv on V (matches RALA's lepe)
        self.lepe = nn.Conv2d(d_model, d_model, 5, 1, 2, groups=d_model)
        # Output projection (matches RALA's self.proj)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _pad_to_chunk(self, x, w_gating):
        """Pad sequence length to be a multiple of chunk_size."""
        b, n, d = x.shape
        C = self.chunk_size
        remainder = n % C
        if remainder == 0:
            return x, w_gating, n
        pad_len = C - remainder
        x = F.pad(x, (0, 0, 0, pad_len))
        w_gating = F.pad(w_gating, (0, pad_len))
        return x, w_gating, n

    def forward(self, x, w_gating, H, W, use_dilution=False):
        """
        Args:
            x:            (B, N, d_model)
            w_gating:     (B, N) per-token gating weights from the shared AC
            H, W:         spatial dimensions of the patch grid (for LePE)
            use_dilution: bool — enable state dilution Γ_τ
        Returns:
            out:   (B, N, d_model) attention output
            phi_k: for Fisher info computation
        """
        # Pad to multiple of chunk_size
        x, w_gating, orig_n = self._pad_to_chunk(x, w_gating)
        b, n, d = x.shape
        T = n // self.chunk_size
        C = self.chunk_size

        # Fused Q/K/V/O projection (matches RALA's self.qkvo)
        qkvo = self.qkvo(x)                          # (B, N, 4*d_model)
        qkv = qkvo[:, :, :3 * self.d_model]           # (B, N, 3*d_model)
        o = qkvo[:, :, 3 * self.d_model:]              # (B, N, d_model) — output gate

        # LePE: 5×5 depthwise conv on V (matches RALA's self.lepe)
        # V is the last d_model slice of qkv
        v_for_lepe = qkv[:, :orig_n, 2 * self.d_model:]  # (B, orig_n, d_model)
        lepe = v_for_lepe.transpose(1, 2).view(b, self.d_model, H, W)  # (B, C, H, W)
        lepe = self.lepe(lepe)                         # (B, C, H, W)
        lepe = lepe.view(b, self.d_model, -1).transpose(1, 2)  # (B, H*W, d_model)

        # Split into heads for chunkwise recurrence
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, N, d_model)
        q = rearrange(q, 'b (T C) (h dk) -> b T h C dk',
                       T=T, C=C, h=self.head)
        k = rearrange(k, 'b (T C) (h dk) -> b T h C dk',
                       T=T, C=C, h=self.head)
        v = rearrange(v, 'b (T C) (h dk) -> b T h C dk',
                       T=T, C=C, h=self.head)

        q = q * (self.d_k ** -0.25)
        k = k * (self.d_k ** -0.25)

        phi_q = F.elu(q) + 1.0
        phi_k = F.elu(k) + 1.0

        w_chunks = rearrange(w_gating, 'b (T C) -> b T C', T=T, C=C)
        w_expanded = w_chunks.unsqueeze(2).unsqueeze(-1)
        k_gated = w_expanded * phi_k

        k_gated_f32 = k_gated.to(torch.float32)
        v_f32 = v.to(torch.float32)
        KV_chunks = torch.matmul(k_gated_f32.transpose(-2, -1), v_f32)
        Z_chunks = k_gated_f32.sum(dim=-2)
        w_bar = w_chunks.mean(dim=-1)

        outputs = []
        S = torch.zeros(b, self.head, self.d_k, self.d_k,
                        device=x.device, dtype=torch.float32)
        Z = torch.zeros(b, self.head, self.d_k,
                        device=x.device, dtype=torch.float32)

        for t in range(T):
            decay_factor = 1.0 - (self.gamma * (1.0 - w_bar[:, t]))
            decay_S = decay_factor.view(b, 1, 1, 1)
            decay_Z = decay_factor.view(b, 1, 1)

            if use_dilution and t > 0:
                dilution_scale = self.gamma * (1.0 - w_bar[:, t])
                gamma_tau = dilution_scale.view(b, 1, 1, 1) * S / max(t, 1)
                S = (S * decay_S) + KV_chunks[:, t] + gamma_tau
            else:
                S = (S * decay_S) + KV_chunks[:, t]

            Z = (Z * decay_Z) + Z_chunks[:, t]
            phi_q_t = phi_q[:, t].to(torch.float32)
            nom = torch.matmul(phi_q_t, S)
            denom = (phi_q_t * Z.unsqueeze(-2)).sum(dim=-1, keepdim=True) + 1e-5
            out_t = nom / denom

            if self.training:
                out_t = self.dropout(out_t)
            out_t = torch.clamp(out_t, min=-65000.0, max=65000.0)
            outputs.append(out_t.to(q.dtype))

        out = torch.stack(outputs, dim=1)
        out = rearrange(out, 'b T h C dk -> b (T C) (h dk)')

        # Remove padding from attention output, then add LePE
        out = out[:, :orig_n, :]
        out = out + lepe

        # RALA-style output gate: raw multiply (no sigmoid) + projection
        o = o[:, :orig_n, :]
        out = self.proj(out * o)

        return out, phi_k


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class ConditionalPositionEncoding(nn.Module):
    """CPE: 3×3 depthwise conv with residual connection (from RAVLT/CPVT)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.dwconv = nn.Conv2d(
            d_model, d_model, kernel_size=3, padding=1, groups=d_model
        )

    def forward(self, x, H, W):
        """
        Args:
            x: (B, N, C) where N = H * W
            H, W: spatial dimensions
        Returns:
            x + CPE: (B, N, C)
        """
        B, N, C = x.shape
        feat = x.transpose(1, 2).view(B, C, H, W)
        feat = self.dwconv(feat)
        feat = feat.view(B, C, N).transpose(1, 2)
        return x + feat


class EncoderBlock(nn.Module):
    """
    A single encoder block. The attention module does NOT own a router.
    The router (SharedActorCritic) lives at the backbone level.
    Includes CPE for positional information.
    """

    def __init__(self, d_model, head, chunk_size, drop_path=0.0):
        super().__init__()
        self.cpe = ConditionalPositionEncoding(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = ChunkwiseRALAAttention(d_model, head=head, chunk_size=chunk_size)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_model * 4)

        # Stochastic depth / DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, w_gating, H, W, use_dilution=False):
        """
        Args:
            x: (B, N, d_model)
            w_gating: (B, N) gating weights
            H, W: spatial dimensions for CPE
            use_dilution: enable state dilution
        Returns:
            x: (B, N, d_model)
            phi_k: for Fisher info
        """
        x = self.cpe(x, H, W)
        res = x
        x_normed = self.norm1(x)
        out, phi_k = self.attn(x_normed, w_gating, H, W, use_dilution)
        x = res + self.drop_path(out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, phi_k


class PatchEmbedding(nn.Module):
    """Patch embedding via strided Conv2d. Handles variable input sizes."""

    def __init__(self, patch_size=16, in_chans=3, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            tokens: (B, H_p * W_p, embed_dim)
            H_p, W_p: patch grid dimensions
        """
        x = self.proj(x)  # (B, embed_dim, H_p, W_p)
        B, C, H_p, W_p = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H_p*W_p, embed_dim)
        return x, H_p, W_p


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor
