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
DEUS Image-to-Image Pipeline.

This pipeline converts an input image to a new image based on a text prompt,
using VAE encoding + noise addition (same approach as SDXL Img2Img).

Key differences from DeusPipeline (text-to-image):
1. Takes an input image instead of starting from pure noise
2. Uses strength parameter to control transformation amount
3. Encodes input image via VAE and adds noise based on strength
"""

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
        >>> from diffusers import DeusImg2ImgPipeline
        >>> from diffusers.utils import load_image

        >>> pipe = DeusImg2ImgPipeline.from_pretrained("path/to/deus-model", torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> init_image = load_image("path/to/image.png")
        >>> prompt = "a beautiful anime girl with long hair, masterpiece"
        >>> image = pipe(prompt, image=init_image, strength=0.8).images[0]
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


class DeusImg2ImgPipeline(DiffusionPipeline):
    """
    Pipeline for image-to-image generation using DEUS architecture.

    This pipeline uses:
    - VAE to encode input image to latent space
    - Noise addition based on strength parameter
    - SigLIP-2 as text encoder
    - DEUS U-Net with RoPE 2D positional encoding
    - 2-Pass CFG for inference

    Args:
        vae (`AutoencoderKL`):
            VAE model for encoding/decoding images (same as SDXL).
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
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

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

    def prepare_latents(
        self,
        image: torch.Tensor,
        timestep: torch.Tensor,
        batch_size: int,
        num_images_per_prompt: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> torch.Tensor:
        """
        Encode image to latents and add noise.
        """
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == 4:
            # Already latents
            init_latents = image
        else:
            # Encode with VAE
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            if isinstance(generator, list):
                init_latents = [
                    retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                    for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = retrieve_latents(self.vae.encode(image), generator=generator)

            init_latents = self.vae.config.scaling_factor * init_latents

        # Duplicate for batch
        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            init_latents = torch.cat([init_latents] * (batch_size // init_latents.shape[0]), dim=0)
        elif batch_size > init_latents.shape[0]:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )

        # Add noise
        shape = init_latents.shape
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)

        return init_latents

    def check_inputs(
        self,
        prompt: Union[str, List[str]],
        image: PipelineImageInput,
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
        strength: float = 0.8,
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
    ) -> Union[DeusPipelineOutput, Tuple]:
        r"""
        Image-to-image generation with DEUS.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation.
            image (`torch.Tensor` or `PIL.Image.Image` or `List`):
                The input image(s) to transform.
            strength (`float`, *optional*, defaults to 0.8):
                How much to transform the input image. Higher values mean more transformation.
                Must be between 0 and 1.
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

        Examples:

        Returns:
            [`DeusPipelineOutput`] or `tuple`:
                If `return_dict` is True, [`DeusPipelineOutput`] is returned,
                otherwise a tuple with generated images.
        """
        # 1. Check inputs
        self.check_inputs(
            prompt, image, strength, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
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

        # 4. Preprocess image
        image = self.image_processor.preprocess(image)

        # 5. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        self._num_timesteps = len(timesteps)

        # 6. Prepare latents
        if latents is None:
            latents = self.prepare_latents(
                image,
                latent_timestep,
                batch_size,
                num_images_per_prompt,
                prompt_embeds.dtype,
                device,
                generator,
            )

        # 7. Denoising loop
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

                # Callback
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 8. Post-processing
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
