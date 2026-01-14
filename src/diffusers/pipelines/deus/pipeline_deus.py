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
DEUS Pipeline for text-to-image and image-conditioned generation.

Key differences from SDXL:
1. Uses SigLIP-2 as text encoder (1152d) instead of dual CLIP (2048d)
2. No time_ids or pooled embeddings
3. Supports variable sequence length text embeddings
4. 2-Pass CFG for inference due to variable sequence lengths
5. RoPE 2D positional encoding in the U-Net
"""

import inspect
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from safetensors.torch import load_file
from transformers import SiglipModel, SiglipProcessor, Siglip2Model, Siglip2TextModel, AutoTokenizer

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...image_processor import PipelineImageInput, VaeImageProcessor
from ...models import AutoencoderKL
from ...models.unets.unet_2d_condition_deus import DeusUNet2DConditionModel
from ...schedulers import KarrasDiffusionSchedulers, EulerDiscreteScheduler
from ...utils import logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import DeusPipelineOutput


logger = logging.get_logger(__name__)

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import DeusPipeline

        >>> pipe = DeusPipeline.from_pretrained("path/to/deus-model", torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a beautiful anime girl with long hair"
        >>> image = pipe(prompt).images[0]
        ```
"""


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale noise prediction for better image quality.
    Based on Section 3.4 from Common Diffusion Noise Schedules paper.
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Get timesteps from scheduler.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")

    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps

    return timesteps, num_inference_steps


class DeusPipeline(DiffusionPipeline):
    """
    Pipeline for text-to-image generation using DEUS architecture.

    This pipeline uses:
    - SigLIP-2 as text encoder (supports both text-only and image-conditioned modes)
    - DEUS U-Net with RoPE 2D positional encoding
    - SDXL VAE for encoding/decoding
    - 2-Pass CFG for inference (due to variable sequence lengths)

    Args:
        vae (`AutoencoderKL`):
            VAE model for encoding/decoding images (same as SDXL).
        text_encoder (`SiglipModel`):
            SigLIP-2 model for text encoding.
        processor (`SiglipProcessor`):
            SigLIP-2 processor for tokenization and image preprocessing.
        unet (`DeusUNet2DConditionModel`):
            DEUS U-Net model with RoPE 2D support.
        scheduler (`KarrasDiffusionSchedulers`):
            Diffusion scheduler (DDPM, DPM++, Euler, etc.).
        tokenizer_max_length (`int`, *optional*, defaults to None):
            Maximum token length for text encoding. If None, no truncation is applied
            (uses the full prompt length). SigLIP-2 supports variable length inputs.
    """

    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    # Key prefixes for single file loading (DEUS format)
    UNET_PREFIX = "model.diffusion_model."
    VAE_PREFIX = "first_stage_model."
    TEXT_ENCODER_PREFIX = "conditioner.embedders.0.model."

    @classmethod
    def from_single_file(
        cls,
        pretrained_model_link_or_path: str,
        *,
        torch_dtype: Optional[torch.dtype] = None,
        text_encoder_config: Optional[Dict[str, Any]] = None,
        vae_config: Optional[Dict[str, Any]] = None,
        unet_config: Optional[Dict[str, Any]] = None,
        scheduler: Optional[KarrasDiffusionSchedulers] = None,
        tokenizer_pretrained: Optional[str] = None,
        **kwargs,
    ):
        """
        Load a DEUS pipeline from a single safetensors file.

        The safetensors file should contain all model components with the following prefixes:
        - UNet: "model.diffusion_model.*"
        - VAE: "first_stage_model.*"
        - Text Encoder: "conditioner.embedders.0.model.*"

        Args:
            pretrained_model_link_or_path (`str`):
                Path to the safetensors file containing all model weights.
            torch_dtype (`torch.dtype`, *optional*):
                The torch dtype to load the model with.
            text_encoder_config (`Dict`, *optional*):
                Configuration for the text encoder. If not provided, defaults are used.
            vae_config (`Dict`, *optional*):
                Configuration for the VAE. If not provided, defaults are used.
            unet_config (`Dict`, *optional*):
                Configuration for the UNet. If not provided, defaults are used.
            scheduler (`KarrasDiffusionSchedulers`, *optional*):
                Scheduler to use. If not provided, EulerDiscreteScheduler is used.
            tokenizer_pretrained (`str`, *optional*):
                Path or repo id for the tokenizer. If not provided, uses SigLIP-2 SO400M tokenizer.

        Returns:
            `DeusPipeline`: The loaded pipeline.

        Example:
            ```python
            from diffusers import DeusPipeline

            pipe = DeusPipeline.from_single_file(
                "path/to/deus_full.safetensors",
                torch_dtype=torch.float16,
            )
            pipe = pipe.to("cuda")
            ```
        """
        # Load checkpoint
        if pretrained_model_link_or_path.endswith(".safetensors"):
            checkpoint = load_file(pretrained_model_link_or_path)
        else:
            raise ValueError("Only .safetensors files are supported for from_single_file")

        # Split state dict by component
        unet_state_dict = {}
        vae_state_dict = {}
        text_encoder_state_dict = {}

        for key, value in checkpoint.items():
            if key.startswith(cls.UNET_PREFIX):
                new_key = key[len(cls.UNET_PREFIX):]
                unet_state_dict[new_key] = value
            elif key.startswith(cls.VAE_PREFIX):
                new_key = key[len(cls.VAE_PREFIX):]
                vae_state_dict[new_key] = value
            elif key.startswith(cls.TEXT_ENCODER_PREFIX):
                new_key = key[len(cls.TEXT_ENCODER_PREFIX):]
                text_encoder_state_dict[new_key] = value

        logger.info(
            f"Loaded checkpoint with {len(unet_state_dict)} UNet keys, "
            f"{len(vae_state_dict)} VAE keys, {len(text_encoder_state_dict)} text encoder keys"
        )

        # Create UNet
        if unet_config is None:
            unet_config = {}
        unet = DeusUNet2DConditionModel(**unet_config)
        unet.load_state_dict(unet_state_dict)
        if torch_dtype is not None:
            unet = unet.to(torch_dtype)
        logger.info(f"Loaded UNet with {sum(p.numel() for p in unet.parameters()):,} parameters")

        # Create VAE
        if vae_config is None:
            vae_config = {
                "in_channels": 3,
                "out_channels": 3,
                "down_block_types": ("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
                "up_block_types": ("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
                "block_out_channels": (128, 256, 512, 512),
                "layers_per_block": 2,
                "latent_channels": 4,
            }
        vae = AutoencoderKL(**vae_config)
        vae.load_state_dict(vae_state_dict)
        if torch_dtype is not None:
            vae = vae.to(torch_dtype)
        logger.info(f"Loaded VAE with {sum(p.numel() for p in vae.parameters()):,} parameters")

        # Create Text Encoder (SigLIP-2)
        # Infer config from state dict if not provided
        if text_encoder_config is None:
            # Try to infer config from weights
            # Look for position embedding to determine max_position_embeddings
            pos_emb_key = "text_model.embeddings.position_embedding.weight"
            if pos_emb_key in text_encoder_state_dict:
                max_pos_emb = text_encoder_state_dict[pos_emb_key].shape[0]
                hidden_size = text_encoder_state_dict[pos_emb_key].shape[1]
            else:
                max_pos_emb = 512
                hidden_size = 1152

            from transformers import Siglip2TextConfig
            text_encoder_config = Siglip2TextConfig(
                hidden_size=hidden_size,
                intermediate_size=4304,
                num_hidden_layers=27,
                num_attention_heads=16,
                max_position_embeddings=max_pos_emb,
                vocab_size=256000,
                hidden_act="gelu_pytorch_tanh",
                layer_norm_eps=1e-6,
            )

        text_encoder = Siglip2TextModel(text_encoder_config)
        text_encoder.load_state_dict(text_encoder_state_dict)
        if torch_dtype is not None:
            text_encoder = text_encoder.to(torch_dtype)
        logger.info(f"Loaded Text Encoder with {sum(p.numel() for p in text_encoder.parameters()):,} parameters")

        # Load tokenizer
        if tokenizer_pretrained is None:
            tokenizer_pretrained = "google/siglip2-so400m-patch16-naflex"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_pretrained)

        # Create processor (for compatibility)
        processor = SiglipProcessor.from_pretrained(tokenizer_pretrained)

        # Create scheduler
        if scheduler is None:
            scheduler = EulerDiscreteScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                num_train_timesteps=1000,
            )

        # Create pipeline
        pipe = cls(
            vae=vae,
            text_encoder=text_encoder,
            processor=processor,
            unet=unet,
            scheduler=scheduler,
            **kwargs,
        )

        return pipe

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: SiglipModel,
        processor: SiglipProcessor,
        unet: DeusUNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        tokenizer_max_length: Optional[int] = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            processor=processor,
            unet=unet,
            scheduler=scheduler,
        )

        # VAE scaling factor (same as SDXL)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        # Default configuration
        self.default_sample_size = self.unet.config.sample_size

        # Tokenizer max length configuration
        # Note: SigLIP-2's default max_position_embeddings is 64, which is very short.
        # For DEUS, the text encoder should be configured with a larger max_position_embeddings
        # (e.g., 256 or 512) to support longer prompts.
        # If tokenizer_max_length is None, we use the model's max_position_embeddings as the limit.
        if tokenizer_max_length is None:
            # Get max_position_embeddings from text encoder config
            text_config = getattr(text_encoder, "text_model", text_encoder).config
            self.tokenizer_max_length = getattr(text_config, "max_position_embeddings", 64)
            logger.info(
                f"tokenizer_max_length not specified, using text encoder's "
                f"max_position_embeddings: {self.tokenizer_max_length}"
            )
        else:
            self.tokenizer_max_length = tokenizer_max_length

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        condition_image: Optional[PipelineImageInput] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode prompts using SigLIP-2.

        Args:
            prompt: Text prompt(s) to encode.
            device: Device for encoding.
            num_images_per_prompt: Number of images per prompt.
            do_classifier_free_guidance: Whether to use CFG.
            negative_prompt: Negative prompt(s).
            prompt_embeds: Pre-computed prompt embeddings.
            negative_prompt_embeds: Pre-computed negative prompt embeddings.
            condition_image: Optional conditioning image for image-conditioned mode.

        Returns:
            Tuple of (prompt_embeds, negative_prompt_embeds).
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Encode positive prompt
        if prompt_embeds is None:
            prompt_embeds = self._encode_single_prompt(
                prompt,
                device,
                condition_image=condition_image,
            )

        # Duplicate for num_images_per_prompt
        prompt_embeds = prompt_embeds.repeat(num_images_per_prompt, 1, 1)

        # Encode negative prompt for CFG
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size

            negative_prompt_embeds = self._encode_single_prompt(
                negative_prompt,
                device,
                condition_image=None,  # No conditioning image for negative
            )

        if do_classifier_free_guidance:
            negative_prompt_embeds = negative_prompt_embeds.repeat(num_images_per_prompt, 1, 1)

        return prompt_embeds, negative_prompt_embeds

    def _encode_single_prompt(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        condition_image: Optional[PipelineImageInput] = None,
    ) -> torch.Tensor:
        """
        Encode a single prompt using SigLIP-2 with variable length support.

        SigLIP-2 natively supports variable length inputs. This method:
        - Uses no truncation by default (tokenizer_max_length=None)
        - Pads to the longest sequence in the batch when batch_size > 1
        - For single prompts, no padding is needed

        Args:
            prompt: Text prompt(s).
            device: Device for encoding.
            condition_image: Optional conditioning image.

        Returns:
            Encoded prompt embeddings of shape (batch, seq_len, 1152).
            seq_len varies based on the actual prompt length.
        """
        if isinstance(prompt, str):
            prompt = [prompt]

        # Configure tokenization for variable length support
        # DEUS supports variable length text inputs without truncation.
        # The text encoder's max_position_embeddings should be configured
        # appropriately (e.g., 256 or 512) to support longer prompts.
        #
        # Strategy:
        # - No truncation (full prompt length preserved)
        # - Use "longest" padding for batches to minimize padding tokens
        # - Use no padding for single prompts (saves computation)
        tokenizer_kwargs = {
            "return_tensors": "pt",
            "padding": "longest" if len(prompt) > 1 else False,
        }

        # Process inputs
        if condition_image is not None:
            # Image-conditioned mode: text + image patches
            inputs = self.processor(
                text=prompt,
                images=condition_image,
                **tokenizer_kwargs,
            ).to(device)

            # Get embeddings from text encoder
            with torch.no_grad():
                outputs = self.text_encoder(**inputs)
                # Use last_hidden_state from text model
                # Note: SigLIP-2 doesn't return hidden_states even with output_hidden_states=True
                prompt_embeds = outputs.text_model_output.last_hidden_state
        else:
            # Text-only mode
            # Access tokenizer directly for more control
            text_inputs = self.processor.tokenizer(
                prompt,
                **tokenizer_kwargs,
            )
            input_ids = text_inputs.input_ids.to(device)
            attention_mask = text_inputs.attention_mask.to(device) if "attention_mask" in text_inputs else None

            # Get embeddings from text encoder
            with torch.no_grad():
                outputs = self.text_encoder.text_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                # Use last_hidden_state (final layer output)
                # Note: SigLIP-2 doesn't return hidden_states even with output_hidden_states=True
                prompt_embeds = outputs.last_hidden_state

        return prompt_embeds

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Prepare latents for denoising.

        Args:
            batch_size: Batch size.
            num_channels_latents: Number of latent channels (4 for VAE).
            height: Target image height.
            width: Target image width.
            dtype: Data type.
            device: Device.
            generator: Random generator.
            latents: Pre-computed latents.

        Returns:
            Initial latents tensor.
        """
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # Scale by scheduler's init noise sigma
        latents = latents * self.scheduler.init_noise_sigma

        return latents

    def check_inputs(
        self,
        prompt: Union[str, List[str]],
        height: int,
        width: int,
        callback_steps: int,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
    ):
        """Validate inputs."""
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"Height and width must be divisible by 8. Got height={height}, width={width}.")

        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(f"callback_steps must be a positive integer, got {callback_steps}.")

        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Cannot pass both prompt and prompt_embeds.")

        if prompt is None and prompt_embeds is None:
            raise ValueError("Must pass either prompt or prompt_embeds.")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError("Cannot pass both negative_prompt and negative_prompt_embeds.")

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        condition_image: Optional[PipelineImageInput] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        guidance_rescale: float = 0.0,
    ) -> Union[DeusPipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings.
            condition_image (`PipelineImageInput`, *optional*):
                Image for SigLIP-2 image-conditioned mode.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DeusPipelineOutput`] instead of a plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called.
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf).

        Examples:

        Returns:
            [`DeusPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`DeusPipelineOutput`] is returned, otherwise a `tuple` is returned
                where the first element is a list with the generated images.
        """
        # 0. Default height and width
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        self._guidance_scale = guidance_scale

        # 2. Determine batch size
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode prompts
        # Note: DEUS uses 2-pass CFG because sequence lengths differ
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            condition_image=condition_image,
        )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # DEUS uses 2-Pass CFG due to variable sequence lengths
                if do_classifier_free_guidance:
                    # Pass 1: Positive embeddings
                    noise_pred_pos = self.unet(
                        latents,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        return_dict=False,
                    )[0]

                    # Pass 2: Negative embeddings
                    noise_pred_neg = self.unet(
                        latents,
                        t,
                        encoder_hidden_states=negative_prompt_embeds,
                        return_dict=False,
                    )[0]

                    # Combine with CFG
                    noise_pred = noise_pred_neg + guidance_scale * (noise_pred_pos - noise_pred_neg)

                    # Apply guidance rescale if needed
                    if guidance_rescale > 0.0:
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_pos, guidance_rescale)
                else:
                    # Single pass without CFG
                    noise_pred = self.unet(
                        latents,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        return_dict=False,
                    )[0]

                # Scheduler step
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # Callback
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 7. Post-processing
        if output_type == "latent":
            image = latents
        else:
            # Decode latents
            latents = latents / self.vae.config.scaling_factor
            image = self.vae.decode(latents, return_dict=False)[0]

            # Post-process
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return DeusPipelineOutput(images=image)
