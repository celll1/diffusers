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
DEUS Inpainting Pipeline.

This pipeline performs mask-based inpainting using VAE encoding + masked latent conditioning.
Follows the same pattern as SDXL Inpaint.

Key features:
1. Takes an input image and mask
2. Encodes input image via VAE
3. Uses mask to blend original and generated content
4. Supports variable strength for partial inpainting
"""

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
from transformers import SiglipModel, SiglipProcessor

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
        >>> from diffusers import DeusInpaintPipeline
        >>> from diffusers.utils import load_image

        >>> pipe = DeusInpaintPipeline.from_pretrained("path/to/deus-model", torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> init_image = load_image("path/to/image.png")
        >>> mask_image = load_image("path/to/mask.png")
        >>> prompt = "a beautiful flower garden"
        >>> image = pipe(prompt, image=init_image, mask_image=mask_image, strength=0.8).images[0]
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


def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    """
    Retrieve latents from VAE encoder output.
    """
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


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


def prepare_mask_and_masked_image(image, mask, height, width, return_image=False):
    """
    Prepare mask and masked image for inpainting.

    Args:
        image: Input image (PIL, np.array, or torch.Tensor).
        mask: Mask image (PIL, np.array, or torch.Tensor).
        height: Target height.
        width: Target width.
        return_image: Whether to return the processed image.

    Returns:
        Tuple of (mask, masked_image) or (mask, masked_image, image).
    """
    if isinstance(image, torch.Tensor):
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.to(dtype=torch.float32)
    elif isinstance(image, PIL.Image.Image):
        image = image.resize((width, height), PIL.Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
    elif isinstance(image, np.ndarray):
        image = image.astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)

    if isinstance(mask, torch.Tensor):
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3:
            mask = mask.unsqueeze(0)
        mask = mask.to(dtype=torch.float32)
    elif isinstance(mask, PIL.Image.Image):
        mask = mask.resize((width, height), PIL.Image.LANCZOS)
        mask = np.array(mask.convert("L")).astype(np.float32) / 255.0
        mask = mask[None, None]
        mask = torch.from_numpy(mask)
    elif isinstance(mask, np.ndarray):
        mask = mask.astype(np.float32) / 255.0
        if mask.ndim == 2:
            mask = mask[None, None]
        elif mask.ndim == 3:
            mask = mask[None]
        mask = torch.from_numpy(mask)

    # Binarize mask
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1

    # Create masked image (mask=1 means area to inpaint)
    masked_image = image * (1 - mask)

    if return_image:
        return mask, masked_image, image

    return mask, masked_image


class DeusInpaintPipeline(DiffusionPipeline):
    """
    Pipeline for inpainting using DEUS architecture.

    This pipeline uses:
    - VAE to encode input image to latent space
    - Mask-based latent conditioning
    - SigLIP-2 as text encoder
    - DEUS U-Net with RoPE 2D positional encoding
    - 2-Pass CFG for inference

    Args:
        vae (`AutoencoderKL`):
            VAE model for encoding/decoding images.
        text_encoder (`SiglipModel`):
            SigLIP-2 model for text encoding.
        processor (`SiglipProcessor`):
            SigLIP-2 processor for tokenization.
        unet (`DeusUNet2DConditionModel`):
            DEUS U-Net model with RoPE 2D support.
        scheduler (`KarrasDiffusionSchedulers`):
            Diffusion scheduler.
    """

    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds", "mask", "masked_image_latents"]

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

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )

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
        Encode prompts using SigLIP-2.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Encode positive prompt
        if prompt_embeds is None:
            prompt_embeds = self._encode_single_prompt(prompt, device)

        prompt_embeds = prompt_embeds.repeat(num_images_per_prompt, 1, 1)

        # Encode negative prompt for CFG
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size

            negative_prompt_embeds = self._encode_single_prompt(negative_prompt, device)

        if do_classifier_free_guidance:
            negative_prompt_embeds = negative_prompt_embeds.repeat(num_images_per_prompt, 1, 1)

        return prompt_embeds, negative_prompt_embeds

    def _encode_single_prompt(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Encode a single prompt using SigLIP-2.
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

    def _encode_vae_image(
        self,
        image: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Encode image with VAE.
        """
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        image_latents = self.vae.config.scaling_factor * image_latents

        return image_latents

    def get_timesteps(self, num_inference_steps: int, strength: float, device: torch.device):
        """
        Get timesteps based on strength.
        """
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)

        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, num_inference_steps - t_start

    def prepare_mask_latents(
        self,
        mask: torch.Tensor,
        masked_image: torch.Tensor,
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
        do_classifier_free_guidance: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare mask and masked image latents.
        """
        # Resize mask to latent space
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)

        # Duplicate for batch
        if mask.shape[0] < batch_size:
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)

        # Encode masked image
        masked_image = masked_image.to(device=device, dtype=dtype)
        masked_image_latents = self._encode_vae_image(masked_image, generator=generator)

        # Duplicate for batch
        if masked_image_latents.shape[0] < batch_size:
            masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)

        return mask, masked_image_latents

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
        image: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        is_strength_max: bool = True,
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

        # If not starting from pure noise, add noise to image latents
        if not is_strength_max and image is not None:
            image_latents = self._encode_vae_image(image, generator=generator)
            noise = latents
            latents = self.scheduler.add_noise(image_latents, noise, timestep)

        return latents

    def check_inputs(
        self,
        prompt: Union[str, List[str]],
        image: PipelineImageInput,
        mask_image: PipelineImageInput,
        strength: float,
        callback_steps: int,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
    ):
        """Validate inputs."""
        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

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
        image: PipelineImageInput = None,
        mask_image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = 1.0,
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
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        guidance_rescale: float = 0.0,
        padding_mask_crop: Optional[int] = None,
    ) -> Union[DeusPipelineOutput, Tuple]:
        r"""
        Inpainting with DEUS.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation.
            image (`torch.Tensor` or `PIL.Image.Image` or `List`):
                The input image(s) for inpainting.
            mask_image (`torch.Tensor` or `PIL.Image.Image` or `List`):
                The mask image(s). White pixels (value 1) indicate areas to inpaint.
            height (`int`, *optional*):
                Target height. Defaults to image height.
            width (`int`, *optional*):
                Target width. Defaults to image width.
            strength (`float`, *optional*, defaults to 1.0):
                How much to transform the masked region. 1.0 means full inpainting.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Classifier-free guidance scale.
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
            padding_mask_crop (`int`, *optional*):
                Size of padding for mask cropping.

        Examples:

        Returns:
            [`DeusPipelineOutput`] or `tuple`:
                If `return_dict` is True, [`DeusPipelineOutput`] is returned,
                otherwise a tuple with generated images.
        """
        # 1. Check inputs
        self.check_inputs(
            prompt, image, mask_image, strength, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
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

        # 4. Preprocess image and mask
        # Get height and width from image
        if isinstance(image, PIL.Image.Image):
            width = width or image.width
            height = height or image.height
        elif isinstance(image, torch.Tensor):
            width = width or image.shape[-1]
            height = height or image.shape[-2]

        # Process image and mask
        mask, masked_image, init_image = prepare_mask_and_masked_image(
            image, mask_image, height, width, return_image=True
        )

        # 5. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)

        is_strength_max = strength == 1.0
        if is_strength_max:
            latent_timestep = None
        else:
            latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        self._num_timesteps = len(timesteps)

        # 6. Prepare mask and masked image latents
        mask, masked_image_latents = self.prepare_mask_latents(
            mask,
            masked_image,
            batch_size * num_images_per_prompt,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            do_classifier_free_guidance,
        )

        # 7. Prepare latents
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
            init_image.to(device=device, dtype=prompt_embeds.dtype) if not is_strength_max else None,
            latent_timestep,
            is_strength_max,
        )

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # DEUS uses 2-Pass CFG
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

                    if guidance_rescale > 0.0:
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_pos, guidance_rescale)
                else:
                    noise_pred = self.unet(
                        latents,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        return_dict=False,
                    )[0]

                # Scheduler step
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # Apply mask blending
                # Blend generated latents with original latents based on mask
                if i < len(timesteps) - 1:
                    noise_timestep = timesteps[i + 1]
                    init_latents_proper = self.scheduler.add_noise(
                        masked_image_latents, randn_tensor(masked_image_latents.shape, generator=generator, device=device, dtype=latents.dtype),
                        torch.tensor([noise_timestep])
                    )
                    latents = (1 - mask) * init_latents_proper + mask * latents

                # Callback
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 9. Final mask blending
        latents = (1 - mask) * masked_image_latents + mask * latents

        # 10. Post-processing
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
