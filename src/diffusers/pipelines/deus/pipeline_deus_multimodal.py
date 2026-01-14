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
DEUS MultiModal Pipeline.

This is a separate ecosystem from DeusPipeline that uses SigLIP-2 Vision Encoder
for image-conditioned generation. Unlike Img2Img which uses VAE encoding + noise,
this pipeline uses the Vision Encoder to create image embeddings that condition
the generation process similar to text embeddings.

Key features:
1. Uses SigLIP-2 Vision Encoder (not VAE) for image conditioning
2. Combines text and image embeddings for conditioning
3. Generates new images based on reference image features
4. Different from Img2Img - this preserves semantic features, not pixel structure
"""

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import PIL.Image
import torch
from transformers import SiglipModel, Siglip2Model, SiglipProcessor, Siglip2Processor

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...image_processor import PipelineImageInput, VaeImageProcessor
from ...models import AutoencoderKL
from ...models.unets.unet_2d_condition_deus import DeusUNet2DConditionModel
from ...schedulers import KarrasDiffusionSchedulers
from ...utils import logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import DeusPipelineOutput


logger = logging.get_logger(__name__)

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import DeusMultiModalPipeline
        >>> from diffusers.utils import load_image

        >>> pipe = DeusMultiModalPipeline.from_pretrained("path/to/deus-multimodal", torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> reference_image = load_image("path/to/reference.png")
        >>> prompt = "a similar character in a different pose"
        >>> image = pipe(
        ...     prompt=prompt,
        ...     reference_image=reference_image,
        ...     image_guidance_scale=1.0,
        ... ).images[0]
        ```
"""


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale noise prediction for better image quality.
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


class DeusMultiModalPipeline(DiffusionPipeline):
    """
    Pipeline for multimodal (text + image) conditioned generation using DEUS architecture.

    This pipeline uses SigLIP-2's Vision Encoder to extract image features and combines
    them with text embeddings for conditioning. Unlike Img2Img which preserves pixel-level
    structure through VAE encoding, this approach preserves semantic/feature-level information.

    Key differences from DeusImg2ImgPipeline:
    - Uses Vision Encoder instead of VAE for image understanding
    - Reference image affects generation semantically, not structurally
    - Can generate entirely new compositions while preserving style/content features

    Args:
        vae (`AutoencoderKL`):
            VAE model for encoding/decoding images.
        text_encoder (`Siglip2Model`):
            SigLIP-2 model for text and image encoding.
        processor (`Siglip2Processor`):
            SigLIP-2 processor for tokenization and image preprocessing.
        unet (`DeusUNet2DConditionModel`):
            DEUS U-Net model with RoPE 2D support.
        scheduler (`KarrasDiffusionSchedulers`):
            Diffusion scheduler.
        image_projection (`torch.nn.Module`, *optional*):
            Optional projection layer to align image embeddings with text embedding space.
    """

    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = ["image_projection"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds", "image_embeds"]

    # Key prefixes for single file loading (DEUS MultiModal format)
    UNET_PREFIX = "model.diffusion_model."
    VAE_PREFIX = "first_stage_model."
    TEXT_ENCODER_PREFIX = "conditioner.embedders.0.model."
    VISION_ENCODER_PREFIX = "conditioner.embedders.1.model."
    IMAGE_PROJECTION_PREFIX = "conditioner.embedders.1.projection."

    def save_to_single_file(
        self,
        save_path: str,
        safe_serialization: bool = True,
    ):
        """
        Save the DEUS MultiModal pipeline to a single safetensors file.

        The file will contain all model components with the following prefixes:
        - UNet: "model.diffusion_model.*"
        - VAE: "first_stage_model.*"
        - Text Encoder: "conditioner.embedders.0.model.*"
        - Vision Encoder: "conditioner.embedders.1.model.*" (from text_encoder.vision_model)
        - Image Projection: "conditioner.embedders.1.projection.*" (if present)

        Args:
            save_path (`str`):
                Path to save the safetensors file. Should end with `.safetensors`.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to use safetensors format. Only safetensors is supported.

        Example:
            ```python
            from diffusers import DeusMultiModalPipeline

            pipe = DeusMultiModalPipeline.from_pretrained("path/to/deus-multimodal")
            pipe.save_to_single_file("deus_multimodal.safetensors")
            ```
        """
        if not safe_serialization:
            raise ValueError("Only safetensors format is supported for save_to_single_file")

        if not save_path.endswith(".safetensors"):
            save_path = save_path + ".safetensors"

        from safetensors.torch import save_file

        combined_state_dict = {}

        # Save UNet weights
        logger.info("Collecting UNet weights...")
        unet_state = self.unet.state_dict()
        for key, value in unet_state.items():
            combined_state_dict[f"{self.UNET_PREFIX}{key}"] = value

        # Save VAE weights
        logger.info("Collecting VAE weights...")
        vae_state = self.vae.state_dict()
        for key, value in vae_state.items():
            combined_state_dict[f"{self.VAE_PREFIX}{key}"] = value

        # Save Text Encoder weights (text_model part)
        logger.info("Collecting Text Encoder weights...")
        if hasattr(self.text_encoder, "text_model"):
            text_encoder_state = self.text_encoder.text_model.state_dict()
        else:
            text_encoder_state = self.text_encoder.state_dict()
        for key, value in text_encoder_state.items():
            combined_state_dict[f"{self.TEXT_ENCODER_PREFIX}{key}"] = value

        # Save Vision Encoder weights (vision_model part)
        logger.info("Collecting Vision Encoder weights...")
        if hasattr(self.text_encoder, "vision_model"):
            vision_encoder_state = self.text_encoder.vision_model.state_dict()
            for key, value in vision_encoder_state.items():
                combined_state_dict[f"{self.VISION_ENCODER_PREFIX}{key}"] = value

        # Save Image Projection weights (if present)
        if self.image_projection is not None:
            logger.info("Collecting Image Projection weights...")
            projection_state = self.image_projection.state_dict()
            for key, value in projection_state.items():
                combined_state_dict[f"{self.IMAGE_PROJECTION_PREFIX}{key}"] = value

        # Save to file
        logger.info(f"Saving to {save_path}...")
        save_file(combined_state_dict, save_path)

        total_params = sum(p.numel() for p in combined_state_dict.values())
        logger.info(
            f"Saved {len(combined_state_dict)} tensors with {total_params:,} parameters "
            f"to {save_path}"
        )

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: Union[SiglipModel, Siglip2Model],
        processor: Union[SiglipProcessor, Siglip2Processor],
        unet: DeusUNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        image_projection: Optional[torch.nn.Module] = None,
        tokenizer_max_length: Optional[int] = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            processor=processor,
            unet=unet,
            scheduler=scheduler,
            image_projection=image_projection,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.default_sample_size = self.unet.config.sample_size

        # Tokenizer max length configuration
        if tokenizer_max_length is None:
            text_config = getattr(text_encoder, "text_model", text_encoder).config
            self.tokenizer_max_length = getattr(text_config, "max_position_embeddings", 64)
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text prompts using SigLIP-2.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Encode positive prompt
        if prompt_embeds is None:
            prompt_embeds = self._encode_text_prompt(prompt, device)

        prompt_embeds = prompt_embeds.repeat(num_images_per_prompt, 1, 1)

        # Encode negative prompt for CFG
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size

            negative_prompt_embeds = self._encode_text_prompt(negative_prompt, device)

        if do_classifier_free_guidance:
            negative_prompt_embeds = negative_prompt_embeds.repeat(num_images_per_prompt, 1, 1)

        return prompt_embeds, negative_prompt_embeds

    def _encode_text_prompt(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Encode text prompt using SigLIP-2 text model.
        """
        if isinstance(prompt, str):
            prompt = [prompt]

        tokenizer_kwargs = {
            "return_tensors": "pt",
            "padding": "longest" if len(prompt) > 1 else False,
        }

        text_inputs = self.processor.tokenizer(prompt, **tokenizer_kwargs)
        input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device) if "attention_mask" in text_inputs else None

        with torch.no_grad():
            outputs = self.text_encoder.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            prompt_embeds = outputs.last_hidden_state

        return prompt_embeds

    def encode_image(
        self,
        image: PipelineImageInput,
        device: torch.device,
        num_images_per_prompt: int = 1,
    ) -> torch.Tensor:
        """
        Encode reference image using SigLIP-2 Vision Encoder.

        Args:
            image: Reference image(s) for conditioning.
            device: Device for encoding.
            num_images_per_prompt: Number of images per prompt.

        Returns:
            Image embeddings of shape (batch, num_patches, hidden_size).
        """
        if not isinstance(image, list):
            image = [image]

        # Preprocess images using SigLIP-2 processor
        pixel_values = self.processor.image_processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device=device, dtype=self.text_encoder.dtype)

        with torch.no_grad():
            # Get vision encoder outputs
            vision_outputs = self.text_encoder.vision_model(pixel_values=pixel_values)
            image_embeds = vision_outputs.last_hidden_state

            # Apply optional projection
            if self.image_projection is not None:
                image_embeds = self.image_projection(image_embeds)

        # Repeat for num_images_per_prompt
        image_embeds = image_embeds.repeat(num_images_per_prompt, 1, 1)

        return image_embeds

    def combine_embeddings(
        self,
        text_embeds: torch.Tensor,
        image_embeds: Optional[torch.Tensor],
        image_guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Combine text and image embeddings.

        Strategy: Concatenate text and image embeddings along sequence dimension.
        The combined embeddings are then used for cross-attention conditioning.

        Args:
            text_embeds: Text embeddings from SigLIP-2 text encoder.
            image_embeds: Image embeddings from SigLIP-2 vision encoder.
            image_guidance_scale: Scale factor for image embeddings.

        Returns:
            Combined embeddings of shape (batch, text_seq + image_seq, hidden_size).
        """
        if image_embeds is None:
            return text_embeds

        # Scale image embeddings
        if image_guidance_scale != 1.0:
            image_embeds = image_embeds * image_guidance_scale

        # Concatenate along sequence dimension
        combined_embeds = torch.cat([text_embeds, image_embeds], dim=1)

        return combined_embeds

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
        reference_image: Optional[PipelineImageInput] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        guidance_rescale: float = 0.0,
    ) -> Union[DeusPipelineOutput, Tuple]:
        r"""
        Multimodal generation with DEUS using text and reference image conditioning.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation.
            reference_image (`PipelineImageInput`, *optional*):
                Reference image(s) for conditioning via Vision Encoder.
                This affects generation semantically, not structurally.
            height (`int`, *optional*):
                The height in pixels of the generated image.
            width (`int`, *optional*):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Classifier-free guidance scale for text conditioning.
            image_guidance_scale (`float`, *optional*, defaults to 1.0):
                Scale factor for image conditioning. Higher values increase
                the influence of the reference image.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Parameter eta from DDIM paper.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                Random generator(s) for deterministic generation.
            latents (`torch.Tensor`, *optional*):
                Pre-generated latents.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings.
            image_embeds (`torch.Tensor`, *optional*):
                Pre-generated image embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                Output format.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a DeusPipelineOutput.
            callback (`Callable`, *optional*):
                Callback function called during inference.
            callback_steps (`int`, *optional*, defaults to 1):
                Frequency of callback calls.
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor.

        Examples:

        Returns:
            [`DeusPipelineOutput`] or `tuple`:
                If `return_dict` is True, [`DeusPipelineOutput`] is returned,
                otherwise a tuple with generated images.
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
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Encode reference image (if provided)
        if image_embeds is None and reference_image is not None:
            image_embeds = self.encode_image(
                reference_image,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )

        # 5. Combine text and image embeddings
        combined_prompt_embeds = self.combine_embeddings(
            prompt_embeds, image_embeds, image_guidance_scale
        )

        if do_classifier_free_guidance:
            # For negative, only use text embeddings (no image influence)
            # Or use zero image embeddings
            if image_embeds is not None:
                zero_image_embeds = torch.zeros_like(image_embeds)
                combined_negative_embeds = self.combine_embeddings(
                    negative_prompt_embeds, zero_image_embeds, 0.0
                )
            else:
                combined_negative_embeds = negative_prompt_embeds

        # 6. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )
        self._num_timesteps = len(timesteps)

        # 7. Prepare latents
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            combined_prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # DEUS uses 2-Pass CFG
                if do_classifier_free_guidance:
                    # Pass 1: Positive embeddings (text + image)
                    noise_pred_pos = self.unet(
                        latents,
                        t,
                        encoder_hidden_states=combined_prompt_embeds,
                        return_dict=False,
                    )[0]

                    # Pass 2: Negative embeddings (text only or text + zero image)
                    noise_pred_neg = self.unet(
                        latents,
                        t,
                        encoder_hidden_states=combined_negative_embeds,
                        return_dict=False,
                    )[0]

                    # Combine with CFG
                    noise_pred = noise_pred_neg + guidance_scale * (noise_pred_pos - noise_pred_neg)

                    if guidance_rescale > 0.0:
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_pos, guidance_rescale)
                else:
                    noise_pred = self.unet(
                        latents,
                        t,
                        encoder_hidden_states=combined_prompt_embeds,
                        return_dict=False,
                    )[0]

                # Scheduler step
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # Callback
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 9. Post-processing
        if output_type == "latent":
            image = latents
        else:
            latents = latents / self.vae.config.scaling_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return DeusPipelineOutput(images=image)
