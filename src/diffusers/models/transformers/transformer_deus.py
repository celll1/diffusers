# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DEUS Transformer2D Model with RoPE 2D positional encoding.

Key differences from SDXL's Transformer2DModel:
- Uses RoPE 2D (Rotary Position Embedding) for spatial attention
- Cross-attention dimension is 1152 (SigLIP-2) instead of 2048
- Supports variable sequence length cross-attention
- No pooled embeddings or time_ids conditioning
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import logging
from ...utils.torch_utils import maybe_allow_in_graph
from ..attention import AttentionMixin, AttentionModuleMixin, FeedForward
from ..attention_dispatch import dispatch_attention_fn
from ..attention_processor import Attention, AttentionProcessor
from ..embeddings import (
    apply_rotary_emb,
    get_2d_rotary_pos_embed,
)
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import RMSNorm


logger = logging.get_logger(__name__)


class DeusAttnProcessor:
    """
    Attention processor for DEUS that applies RoPE 2D embeddings to self-attention
    and standard cross-attention for text conditioning.
    """

    _attention_backend = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version."
            )

    def __call__(
        self,
        attn: "DeusAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape

        # Self-attention: Q, K, V from hidden_states
        # Cross-attention: Q from hidden_states, K, V from encoder_hidden_states
        is_cross_attention = encoder_hidden_states is not None

        query = attn.to_q(hidden_states)

        if is_cross_attention:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
        else:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        # Reshape to (batch, seq, heads, dim_head) for dispatch_attention_fn
        # Note: dispatch_attention_fn expects (B, S, H, D) format
        query = query.view(batch_size, -1, attn.heads, attn.dim_head)
        key = key.view(batch_size, -1, attn.heads, attn.dim_head)
        value = value.view(batch_size, -1, attn.heads, attn.dim_head)

        # Apply RoPE only to self-attention (spatial positions)
        # RoPE expects (B, H, S, D) format, so transpose before and after
        if image_rotary_emb is not None and not is_cross_attention:
            query = query.transpose(1, 2)  # (B, H, S, D)
            key = key.transpose(1, 2)
            query = apply_rotary_emb(query, image_rotary_emb, use_real=True)
            key = apply_rotary_emb(key, image_rotary_emb, use_real=True)
            query = query.transpose(1, 2)  # (B, S, H, D)
            key = key.transpose(1, 2)

        # Compute attention using dispatch_attention_fn
        # Input: (B, S, H, D), Output: (B, S, H, D)
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
        )

        # Reshape back: flatten heads and dim_head
        hidden_states = hidden_states.flatten(2, 3)  # (B, S, H*D)
        hidden_states = hidden_states.to(query.dtype)

        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


@maybe_allow_in_graph
class DeusAttention(nn.Module, AttentionModuleMixin):
    """
    Attention module for DEUS with support for RoPE 2D positional embeddings.

    Args:
        query_dim: The number of channels in the query.
        cross_attention_dim: The number of channels in the encoder hidden states.
                            If None, self-attention is used.
        heads: The number of heads to use for multi-head attention.
        dim_head: The number of channels in each head.
        dropout: The dropout probability to use.
        bias: Whether to use bias in projections.
    """

    _default_processor_cls = DeusAttnProcessor
    _supports_qkv_fusion = False

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)

        self.to_out = nn.ModuleList([
            nn.Linear(self.inner_dim, query_dim, bias=bias),
            nn.Dropout(dropout)
        ])

        self.processor = DeusAttnProcessor()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )


@maybe_allow_in_graph
class DeusBasicTransformerBlock(nn.Module):
    """
    A basic transformer block for DEUS with RoPE 2D support.

    This block contains:
    1. Self-attention with RoPE 2D positional encoding
    2. Cross-attention with text embeddings (no RoPE)
    3. Feed-forward network

    Args:
        dim: The number of channels in the input and output.
        num_attention_heads: The number of heads for multi-head attention.
        attention_head_dim: The dimension of each attention head.
        dropout: The dropout probability.
        cross_attention_dim: The dimension of the cross attention features.
        activation_fn: The activation function to use in the feed-forward network.
        attention_bias: Whether to use bias in attention projections.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        norm_eps: float = 1e-6,
    ):
        super().__init__()

        # Self-attention with RoPE
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn1 = DeusAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
        )

        # Cross-attention (no RoPE, for text conditioning)
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn2 = DeusAttention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
        )

        # Feed-forward
        self.norm3 = nn.LayerNorm(dim, eps=norm_eps)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # 1. Self-Attention with RoPE
        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1(
            norm_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states = hidden_states + attn_output

        # 2. Cross-Attention (no RoPE)
        if encoder_hidden_states is not None:
            norm_hidden_states = self.norm2(hidden_states)
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + ff_output

        return hidden_states


class DeusTransformer2DModel(nn.Module):
    """
    A 2D Transformer model with RoPE 2D positional encoding for DEUS.

    This is used inside the U-Net blocks to process spatial features with
    attention mechanisms that incorporate RoPE 2D for positional information.

    Key features:
    - RoPE 2D positional encoding for self-attention
    - Variable sequence length cross-attention
    - LayerNorm-based normalization

    Args:
        num_attention_heads: Number of attention heads.
        attention_head_dim: Dimension of each attention head.
        in_channels: Number of input channels.
        num_layers: Number of transformer blocks.
        dropout: Dropout probability.
        cross_attention_dim: Dimension of cross-attention features (1152 for SigLIP-2).
        attention_bias: Whether to use bias in attention projections.
        activation_fn: Activation function for feed-forward.
        norm_num_groups: Number of groups for GroupNorm.
        use_linear_projection: Whether to use linear projection instead of conv.
        norm_eps: Epsilon for layer normalization.
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = 1152,  # SigLIP-2 dimension
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        norm_num_groups: int = 32,
        use_linear_projection: bool = False,
        norm_eps: float = 1e-6,
    ):
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_linear_projection = use_linear_projection
        self.gradient_checkpointing = False

        # Input normalization and projection
        self.norm = nn.GroupNorm(
            num_groups=norm_num_groups,
            num_channels=in_channels,
            eps=norm_eps,
            affine=True
        )

        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, self.inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, self.inner_dim, kernel_size=1, stride=1, padding=0)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            DeusBasicTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                dropout=dropout,
                cross_attention_dim=cross_attention_dim,
                activation_fn=activation_fn,
                attention_bias=attention_bias,
                norm_eps=norm_eps,
            )
            for _ in range(num_layers)
        ])

        # Output projection
        if use_linear_projection:
            self.proj_out = nn.Linear(self.inner_dim, self.out_channels)
        else:
            self.proj_out = nn.Conv2d(self.inner_dim, self.out_channels, kernel_size=1, stride=1, padding=0)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_dict: bool = True,
    ) -> Union[Transformer2DModelOutput, Tuple[torch.Tensor]]:
        """
        Forward pass of the DEUS Transformer2D model.

        Args:
            hidden_states: Input tensor of shape (batch, channels, height, width).
            encoder_hidden_states: Cross-attention context of shape (batch, seq_len, cross_dim).
            attention_mask: Attention mask for self-attention.
            encoder_attention_mask: Attention mask for cross-attention.
            image_rotary_emb: Tuple of (cos, sin) for RoPE 2D.
            return_dict: Whether to return a dataclass or tuple.

        Returns:
            Transformer2DModelOutput or tuple with the output tensor.
        """
        batch_size, channels, height, width = hidden_states.shape
        residual = hidden_states

        # 1. Input normalization and projection
        hidden_states = self.norm(hidden_states)

        if self.use_linear_projection:
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
            hidden_states = self.proj_in(hidden_states)
        else:
            hidden_states = self.proj_in(hidden_states)
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_size, height * width, self.inner_dim)

        # 2. Transformer blocks
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    attention_mask,
                    encoder_attention_mask,
                    image_rotary_emb,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    image_rotary_emb=image_rotary_emb,
                )

        # 3. Output projection
        if self.use_linear_projection:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.reshape(batch_size, height, width, self.out_channels).permute(0, 3, 1, 2)
        else:
            hidden_states = hidden_states.reshape(batch_size, height, width, self.inner_dim).permute(0, 3, 1, 2)
            hidden_states = self.proj_out(hidden_states)

        # 4. Residual connection
        output = hidden_states + residual

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
