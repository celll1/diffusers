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
DEUS UNet2DConditionModel

A U-Net architecture for the DEUS diffusion model with the following key differences from SDXL:
1. No added_cond_kwargs (no time_ids, no pooled embeddings)
2. RoPE 2D positional encoding for spatial attention
3. Cross-attention with 1152d text embeddings (SigLIP-2)
4. Variable sequence length cross-attention support
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin, UNet2DConditionLoadersMixin
from ...loaders.single_file_model import FromOriginalModelMixin
from ...utils import BaseOutput, logging
from ..activations import get_activation
from ..attention import Attention, AttentionMixin
from ..attention_processor import (
    CROSS_ATTENTION_PROCESSORS,
    AttentionProcessor,
    AttnProcessor,
    FusedAttnProcessor2_0,
)
from ..embeddings import (
    TimestepEmbedding,
    Timesteps,
    get_2d_rotary_pos_embed,
)
from ..modeling_utils import ModelMixin
from .unet_2d_blocks_deus import (
    DeusCrossAttnDownBlock2D,
    DeusCrossAttnUpBlock2D,
    DeusDownBlock2D,
    DeusMidBlock2DCrossAttn,
    DeusUpBlock2D,
)


logger = logging.get_logger(__name__)


@dataclass
class DeusUNet2DConditionOutput(BaseOutput):
    """
    The output of [`DeusUNet2DConditionModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input.
    """

    sample: torch.Tensor = None


def get_deus_down_block(
    down_block_type: str,
    num_layers: int,
    transformer_layers_per_block: int,
    in_channels: int,
    out_channels: int,
    temb_channels: int,
    add_downsample: bool,
    resnet_eps: float,
    resnet_act_fn: str,
    resnet_groups: int,
    cross_attention_dim: int,
    num_attention_heads: int,
    downsample_padding: int,
    attention_head_dim: int,
    dropout: float = 0.0,
):
    """Get DEUS down block based on type."""
    if down_block_type == "DeusCrossAttnDownBlock2D":
        return DeusCrossAttnDownBlock2D(
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            downsample_padding=downsample_padding,
            attention_head_dim=attention_head_dim,
            dropout=dropout,
        )
    elif down_block_type == "DeusDownBlock2D":
        return DeusDownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown down block type: {down_block_type}")


def get_deus_up_block(
    up_block_type: str,
    num_layers: int,
    transformer_layers_per_block: int,
    in_channels: int,
    out_channels: int,
    prev_output_channel: int,
    temb_channels: int,
    add_upsample: bool,
    resnet_eps: float,
    resnet_act_fn: str,
    resnet_groups: int,
    cross_attention_dim: int,
    num_attention_heads: int,
    attention_head_dim: int,
    dropout: float = 0.0,
    resolution_idx: Optional[int] = None,
):
    """Get DEUS up block based on type."""
    if up_block_type == "DeusCrossAttnUpBlock2D":
        return DeusCrossAttnUpBlock2D(
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            dropout=dropout,
            resolution_idx=resolution_idx,
        )
    elif up_block_type == "DeusUpBlock2D":
        return DeusUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            dropout=dropout,
            resolution_idx=resolution_idx,
        )
    else:
        raise ValueError(f"Unknown up block type: {up_block_type}")


class DeusUNet2DConditionModel(
    ModelMixin, AttentionMixin, ConfigMixin, FromOriginalModelMixin, UNet2DConditionLoadersMixin, PeftAdapterMixin
):
    """
    A conditional 2D UNet model for DEUS that takes a noisy sample, timestep, and encoder_hidden_states
    and returns a denoised sample.

    Key differences from SDXL UNet:
    - No added_cond_kwargs (no time_ids, no pooled embeddings)
    - Uses RoPE 2D positional encoding for spatial attention
    - Cross-attention with 1152d embeddings (SigLIP-2) instead of 2048d
    - Variable sequence length support for cross-attention

    Parameters:
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample.
        in_channels (`int`, *optional*, defaults to 4): Number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 4): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*):
            The tuple of downsample blocks to use.
        mid_block_type (`str`, *optional*, defaults to `"DeusMidBlock2DCrossAttn"`):
            Block type for middle of UNet.
        up_block_types (`Tuple[str]`, *optional*):
            The tuple of upsample blocks to use.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2): The number of layers per block.
        cross_attention_dim (`int`, *optional*, defaults to 1152):
            The dimension of the cross attention features (SigLIP-2).
        attention_head_dim (`int`, *optional*, defaults to 8): The dimension of the attention heads.
        use_rope_2d (`bool`, *optional*, defaults to True):
            Whether to use RoPE 2D positional encoding.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The theta parameter for RoPE.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["DeusBasicTransformerBlock", "ResnetBlock2D", "DeusCrossAttnUpBlock2D", "DeusUpBlock2D"]
    _skip_layerwise_casting_patterns = ["norm"]

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[Union[int, Tuple[int, int]]] = 128,
        in_channels: int = 4,
        out_channels: int = 4,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str, ...] = (
            "DeusCrossAttnDownBlock2D",
            "DeusCrossAttnDownBlock2D",
            "DeusCrossAttnDownBlock2D",
        ),
        mid_block_type: Optional[str] = "DeusMidBlock2DCrossAttn",
        up_block_types: Tuple[str, ...] = (
            "DeusCrossAttnUpBlock2D",
            "DeusCrossAttnUpBlock2D",
            "DeusCrossAttnUpBlock2D",
        ),
        block_out_channels: Tuple[int, ...] = (320, 640, 1280),
        layers_per_block: Union[int, Tuple[int]] = 2,
        downsample_padding: int = 1,
        dropout: float = 0.0,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: Union[int, Tuple[int]] = 1152,  # SigLIP-2 dimension
        transformer_layers_per_block: Union[int, Tuple[int]] = (1, 2, 10),  # Match SDXL structure
        attention_head_dim: Union[int, Tuple[int]] = (8, 16, 32),  # Must be divisible by 4 for RoPE 2D
        use_rope_2d: bool = True,
        rope_theta: float = 10000.0,
        conv_in_kernel: int = 3,
        conv_out_kernel: int = 3,
    ):
        super().__init__()

        self.sample_size = sample_size
        self.use_rope_2d = use_rope_2d
        self.rope_theta = rope_theta

        # Convert to lists if needed
        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = [attention_head_dim] * len(down_block_types)

        if isinstance(cross_attention_dim, int):
            cross_attention_dim = [cross_attention_dim] * len(down_block_types)

        # Input convolution
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
        )

        # Time embedding
        time_embed_dim = block_out_channels[0] * 4
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
        )

        # Down blocks
        self.down_blocks = nn.ModuleList([])
        output_channel = block_out_channels[0]

        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_deus_down_block(
                down_block_type,
                num_layers=layers_per_block[i],
                transformer_layers_per_block=transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim[i],
                num_attention_heads=output_channel // attention_head_dim[i],
                downsample_padding=downsample_padding,
                attention_head_dim=attention_head_dim[i],
                dropout=dropout,
            )
            self.down_blocks.append(down_block)

        # Mid block
        if mid_block_type == "DeusMidBlock2DCrossAttn":
            self.mid_block = DeusMidBlock2DCrossAttn(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                output_scale_factor=1.0,
                transformer_layers_per_block=transformer_layers_per_block[-1],
                num_attention_heads=block_out_channels[-1] // attention_head_dim[-1],
                cross_attention_dim=cross_attention_dim[-1],
                attention_head_dim=attention_head_dim[-1],
                dropout=dropout,
            )
        else:
            self.mid_block = None

        # Up blocks
        self.up_blocks = nn.ModuleList([])
        self.num_upsamplers = 0

        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attention_head_dim = list(reversed(attention_head_dim))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))
        reversed_transformer_layers_per_block = list(reversed(transformer_layers_per_block))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_deus_up_block(
                up_block_type,
                num_layers=reversed_layers_per_block[i] + 1,
                transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=reversed_cross_attention_dim[i],
                num_attention_heads=output_channel // reversed_attention_head_dim[i],
                attention_head_dim=reversed_attention_head_dim[i],
                dropout=dropout,
                resolution_idx=i,
            )
            self.up_blocks.append(up_block)

        # Output
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps
        )
        self.conv_act = get_activation(act_fn)

        conv_out_padding = (conv_out_kernel - 1) // 2
        self.conv_out = nn.Conv2d(
            block_out_channels[0], out_channels, kernel_size=conv_out_kernel, padding=conv_out_padding
        )

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.
        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor)

    def set_attention_slice(self, slice_size: Union[str, int, List[int]] = "auto"):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module splits the input tensor in slices to compute attention in
        several steps. This is useful for saving some memory in exchange for a small decrease in speed.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, input to the attention heads is halved, so attention is computed in two steps. If
                `"max"`, maximum amount of memory is saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_sliceable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_sliceable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_sliceable_dims(module)

        num_sliceable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_sliceable_layers * [1]

        slice_size = num_sliceable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def enable_freeu(self, s1: float, s2: float, b1: float, b2: float):
        r"""Enables the FreeU mechanism from https://huggingface.co/papers/2309.11497.

        The suffixes after the scaling factors represent the stage blocks where they are being applied.

        Please refer to the [official repository](https://github.com/ChenyangSi/FreeU) for combinations of values that
        are known to work well for different pipelines such as Stable Diffusion v1, v2, and Stable Diffusion XL.

        Args:
            s1 (`float`):
                Scaling factor for stage 1 to attenuate the contributions of the skip features. This is done to
                mitigate the "oversmoothing effect" in the enhanced denoising process.
            s2 (`float`):
                Scaling factor for stage 2 to attenuate the contributions of the skip features. This is done to
                mitigate the "oversmoothing effect" in the enhanced denoising process.
            b1 (`float`): Scaling factor for stage 1 to amplify the contributions of backbone features.
            b2 (`float`): Scaling factor for stage 2 to amplify the contributions of backbone features.
        """
        for i, upsample_block in enumerate(self.up_blocks):
            setattr(upsample_block, "s1", s1)
            setattr(upsample_block, "s2", s2)
            setattr(upsample_block, "b1", b1)
            setattr(upsample_block, "b2", b2)

    def disable_freeu(self):
        """Disables the FreeU mechanism."""
        freeu_keys = {"s1", "s2", "b1", "b2"}
        for i, upsample_block in enumerate(self.up_blocks):
            for k in freeu_keys:
                if hasattr(upsample_block, k) or getattr(upsample_block, k, None) is not None:
                    setattr(upsample_block, k, None)

    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        > [!WARNING] > This API is 🧪 experimental.
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedAttnProcessor2_0())

    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        > [!WARNING] > This API is 🧪 experimental.

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def get_time_embed(
        self, sample: torch.Tensor, timestep: Union[torch.Tensor, float, int]
    ) -> torch.Tensor:
        """Get time embedding from timestep."""
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            is_npu = sample.device.type == "npu"
            if isinstance(timestep, float):
                dtype = torch.float32 if (is_mps or is_npu) else torch.float64
            else:
                dtype = torch.int32 if (is_mps or is_npu) else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)
        return t_emb

    def get_rope_2d_embed(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate RoPE 2D embeddings for the given spatial dimensions.

        Args:
            height: Height of the feature map.
            width: Width of the feature map.
            device: Device to create tensors on.
            dtype: Data type for the tensors.

        Returns:
            Tuple of (cos, sin) tensors for RoPE, or None if use_rope_2d is False.
        """
        if not self.use_rope_2d:
            return None

        # Get attention head dimension (use the first block's configuration)
        attention_head_dim = self.config.attention_head_dim
        if isinstance(attention_head_dim, (list, tuple)):
            attention_head_dim = attention_head_dim[0]

        # Generate 2D rotary embeddings
        # crops_coords: ((start_h, start_w), (end_h, end_w))
        crops_coords = ((0, 0), (height, width))
        grid_size = (height, width)

        freqs_cos, freqs_sin = get_2d_rotary_pos_embed(
            embed_dim=attention_head_dim,
            crops_coords=crops_coords,
            grid_size=grid_size,
            use_real=True,
            device=device,
            output_type="pt",
        )

        return (freqs_cos.to(dtype), freqs_sin.to(dtype))

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[DeusUNet2DConditionOutput, Tuple]:
        """
        The forward pass of DeusUNet2DConditionModel.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with shape `(batch, 4, height, width)`.
            timestep (`torch.Tensor` or `float` or `int`):
                The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states with shape `(batch, seq_len, 1152)`.
                Note: seq_len is variable (7-293 for DEUS).
            attention_mask (`torch.Tensor`, *optional*):
                Attention mask for self-attention.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary for cross-attention.
            encoder_attention_mask (`torch.Tensor`, *optional*):
                Attention mask for cross-attention on encoder_hidden_states.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a DeusUNet2DConditionOutput instead of a tuple.

        Returns:
            DeusUNet2DConditionOutput or tuple with the noise prediction.
        """
        # Get spatial dimensions
        batch_size, channels, height, width = sample.shape

        # Determine upsampling
        default_overall_up_factor = 2 ** self.num_upsamplers
        forward_upsample_size = False
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                forward_upsample_size = True
                break

        # Process attention masks
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Time embedding
        t_emb = self.get_time_embed(sample=sample, timestep=timestep)
        emb = self.time_embedding(t_emb)

        # 2. Input projection
        sample = self.conv_in(sample)

        # 3. Generate RoPE 2D embeddings
        # Note: We generate for the current resolution and pass to blocks
        # Each block will handle its own resolution
        image_rotary_emb = None  # Will be generated per-resolution in blocks

        # Handle cross_attention_kwargs
        if cross_attention_kwargs is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            _ = cross_attention_kwargs.pop("scale", 1.0)

        # 4. Down blocks
        down_block_res_samples = (sample,)
        for i, downsample_block in enumerate(self.down_blocks):
            # Generate RoPE for current resolution
            current_height = sample.shape[2]
            current_width = sample.shape[3]

            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                # Get attention head dim for this block
                attn_head_dim = self.config.attention_head_dim
                if isinstance(attn_head_dim, (list, tuple)):
                    attn_head_dim = attn_head_dim[i]

                image_rotary_emb = self._get_rope_for_resolution(
                    current_height, current_width, attn_head_dim, sample.device, sample.dtype
                )

                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    image_rotary_emb=image_rotary_emb,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 5. Mid block
        if self.mid_block is not None:
            current_height = sample.shape[2]
            current_width = sample.shape[3]
            attn_head_dim = self.config.attention_head_dim
            if isinstance(attn_head_dim, (list, tuple)):
                attn_head_dim = attn_head_dim[-1]

            image_rotary_emb = self._get_rope_for_resolution(
                current_height, current_width, attn_head_dim, sample.device, sample.dtype
            )

            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
                image_rotary_emb=image_rotary_emb,
            )

        # 6. Up blocks
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]

            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                current_height = sample.shape[2]
                current_width = sample.shape[3]

                # Get attention head dim for this block (reversed order)
                reversed_attn_head_dim = list(reversed(
                    self.config.attention_head_dim if isinstance(self.config.attention_head_dim, (list, tuple))
                    else [self.config.attention_head_dim] * len(self.up_blocks)
                ))
                attn_head_dim = reversed_attn_head_dim[i]

                image_rotary_emb = self._get_rope_for_resolution(
                    current_height, current_width, attn_head_dim, sample.device, sample.dtype
                )

                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    image_rotary_emb=image_rotary_emb,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

        # 7. Output
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return DeusUNet2DConditionOutput(sample=sample)

    def _get_rope_for_resolution(
        self,
        height: int,
        width: int,
        attention_head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate RoPE 2D embeddings for a specific resolution.

        Args:
            height: Height of the feature map.
            width: Width of the feature map.
            attention_head_dim: Dimension of attention heads.
            device: Device to create tensors on.
            dtype: Data type for the tensors.

        Returns:
            Tuple of (cos, sin) tensors for RoPE.
        """
        if not self.use_rope_2d:
            return None

        crops_coords = ((0, 0), (height, width))
        grid_size = (height, width)

        freqs_cos, freqs_sin = get_2d_rotary_pos_embed(
            embed_dim=attention_head_dim,
            crops_coords=crops_coords,
            grid_size=grid_size,
            use_real=True,
            device=device,
            output_type="pt",
        )

        return (freqs_cos.to(dtype), freqs_sin.to(dtype))
