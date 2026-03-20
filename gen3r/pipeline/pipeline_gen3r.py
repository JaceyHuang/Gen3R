import os
import inspect
import math
from dataclasses import dataclass
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import BaseOutput, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange

from ..models import (
    AutoTokenizer, 
    CLIPModel,
    WanT5EncoderModel, 
    WanTransformer3DModel,
    VGGT,
)
from ..models import GeometryAdapter, AutoencoderKLWan
from ..utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas
from ..utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from ..utils.common_utils import convert_to_token_list
from ..models.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from ..models.vggt.utils.geometry import unproject_depth_map_to_point_map

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        pass
        ```
"""


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def resize_mask(mask, latent, process_first_frame_only=True):
    latent_size = latent.size()
    batch_size, channels, num_frames, height, width = mask.shape

    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :],
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
        
        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :],
                size=target_size,
                mode='trilinear',
                align_corners=False
            )
            resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(
            mask,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
    return resized_mask


@dataclass
class Gen3RPipelineOutput(BaseOutput):
    r"""
    Output class for CogVideo pipelines.

    Args:
        video (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`.
    """

    point_maps: torch.Tensor = None
    depth_maps: torch.Tensor = None
    rgbs: torch.Tensor = None
    pcds: torch.Tensor = None
    point_masks: torch.Tensor = None
    cameras: list[torch.Tensor] = None


class Gen3RPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-video generation using Wan.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    """

    model_cpu_offload_seq = "text_encoder->clip_image_encoder->transformer->vae"

    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
    ]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: WanT5EncoderModel,
        transformer: WanTransformer3DModel,
        clip_image_encoder: CLIPModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        geo_adapter: GeometryAdapter,
        wan_vae: AutoencoderKLWan,
        vggt: VGGT = None,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer, 
            text_encoder=text_encoder, 
            transformer=transformer, 
            clip_image_encoder=clip_image_encoder, 
            scheduler=scheduler,
            geo_adapter=geo_adapter,
            wan_vae=wan_vae,
            vggt=vggt,
        )

        if vggt is not None:
            self.aggregator_scale_factor_spatial = vggt.aggregator.patch_size
        else:
            self.aggregator_scale_factor_spatial = 14  # hardcoded
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.aggregator_scale_factor_spatial)


    def save_pretrained(self, save_directory, *args, **kwargs):
        """Save pipeline as usual, then persist `vggt` into a subfolder if available."""
        vggt_ref = self.vggt
        self._optional_components += ["vggt"]
        setattr(self, "vggt", None)

        super().save_pretrained(save_directory, *args, **kwargs)  # will skip vggt
        try:
            if hasattr(self, "vggt") and vggt_ref is not None:
                vggt_dir = os.path.join(save_directory, "vggt")
                # Prefer model's own save if provided (ModelHubMixin)
                if hasattr(vggt_ref, "save_pretrained") and callable(vggt_ref.save_pretrained):
                    vggt_ref.save_pretrained(vggt_dir)
        except Exception as e:
            logger.warning(f"Failed to save VGGT: {e}")
        finally:
            setattr(self, "vggt", vggt_ref)
            self._optional_components.remove("vggt")


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """Load pipeline, then try to attach `vggt` from subfolder unless explicitly provided."""
        provided_vggt = kwargs.pop("vggt", None)
        cls._optional_components += ["vggt"]
    
        pipeline = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        if getattr(pipeline, "vggt", None) is None:
            if provided_vggt is not None:
                pipeline.vggt = provided_vggt
            else:
                try:
                    vggt_dir = os.path.join(pretrained_model_name_or_path, "vggt")
                    if os.path.isdir(vggt_dir):
                        pipeline.vggt = VGGT.from_pretrained(vggt_dir).to(pipeline.device, pipeline.dtype)
                        pipeline.register_modules(vggt=pipeline.vggt)
                        pipeline.components["vggt"] = pipeline.vggt
                        pipeline.aggregator_scale_factor_spatial = pipeline.vggt.aggregator.patch_size
                        pipeline.image_processor = VaeImageProcessor(vae_scale_factor=pipeline.aggregator_scale_factor_spatial)
                except Exception as e:
                    logger.warning(f"VGGT not loaded from pipeline dir: {e}")

        return pipeline


    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
        prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return [u[:v] for u, v in zip(prompt_embeds, seq_lens)]

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(
        self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        
        shape = (
            batch_size,
            num_channels_latents,
            (num_frames - 1) // self.geo_adapter.temporal_compression_ratio + 1,
            height // self.geo_adapter.spatial_compression_ratio,
            width // self.geo_adapter.spatial_compression_ratio * 2,
        )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma
        return latents
    

    def encode_control_image(self, control_image):
        
        # encode the image with wan vae
        control_image = rearrange((control_image * 2 - 1).clamp(-1, 1), 'b f c h w -> b c f h w')
        wan_latents = self.wan_vae.encode(control_image).latent_dist.sample()  # [B, 16, f, 70, 70]
        
        # use zeros for geometry conditions
        geo_latents = torch.zeros_like(wan_latents, device=wan_latents.device, dtype=wan_latents.dtype)
        control_latents = torch.cat([wan_latents, geo_latents], dim=-1)  # [B, 16, f, 70, 140]

        return control_latents
    

    def prepare_control_latents(
        self, control_image, masks, dtype, device
    ):
        control_image = control_image.to(device=device, dtype=dtype)
        bs = 1
        control_latents = []
        for i in range(0, control_image.shape[0], bs):
            control_image_bs = control_image[i:i+bs]  # [B, F, 3, H, W]
            control_latents_bs = self.encode_control_image(control_image_bs)  # [B, 16, f, 70, 140]
            control_latents_bs = torch.cat(
                [control_latents_bs, masks[i:i+bs].to(control_latents_bs.device)], dim=1)  # [B, 20, f, 70, 140]
            control_latents.append(control_latents_bs)
        control_latents = torch.cat(control_latents, dim=0)

        return control_latents


    def decode_latents(self, latents: torch.Tensor, min_max_depth_mask=False) -> torch.Tensor:
        wan_latents, geo_latents = latents.chunk(2, dim=-1)  # [B, 16, f, 70, 70]
        vggt_tokens = self.geo_adapter.decode(geo_latents).sample  # [B, 10240, F, 40, 40]
        aggregated_token_list, frames = convert_to_token_list(
            rearrange(vggt_tokens, 'b c f h w -> b f h w c'), 
            self.vggt.aggregator.patch_size
        )
        pose_enc = self.vggt.camera_head(aggregated_token_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, frames.shape[-2:])  # [B, F, 3, 4], [B, F, 3, 3]
        point_maps, _ = self.vggt.point_head(aggregated_token_list, frames, 5)  # [B, F, H, W, 3]
        depth_maps, _ = self.vggt.depth_head(aggregated_token_list, frames, 5)  # [B, F, H, W, 1]
        point_map_by_unprojection, point_masks = unproject_depth_map_to_point_map(
            depth_maps.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0), return_point_masks=True, min_max_depth_mask=min_max_depth_mask)  # [F, H, W, 3], [F, H, W]
        point_map_by_unprojection = torch.from_numpy(point_map_by_unprojection).unsqueeze(0)  # [B, F, H, W, 3]
        point_masks = torch.from_numpy(point_masks).unsqueeze(0)  # [B, F, H, W]
        
        rgbs = self.wan_vae.decode(wan_latents).sample  # [B, 3, F, H, W]
        rgbs = rearrange((rgbs / 2 + 0.5).clamp(0, 1), 'b c f h w -> b f h w c')  # [B, F, H, W, 3]
        
        results = {
            "point_maps": point_maps,
            "depth_maps": depth_maps,
            "rgbs": rgbs,
            "pcds": point_map_by_unprojection,
            "point_masks": point_masks,
            'cameras': [extrinsic, intrinsic],
        }
        return results

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.latte.pipeline_latte.LattePipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 14 != 0 or width % 14 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 14 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 560,
        width: int = 560,
        control_cameras: Union[torch.FloatTensor] = None,
        control_images: Union[torch.FloatTensor] = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "numpy",
        return_dict: bool = False,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        clip_image: torch.FloatTensor = None,
        max_sequence_length: int = 512,
        cfg_skip_ratio: int = None,
        shift: int = 5,
        min_max_depth_mask: bool = False,
    ) -> Union[Gen3RPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.
        Args:

        Examples:

        Returns:

        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        weight_dtype = self.text_encoder.dtype

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            in_prompt_embeds = negative_prompt_embeds + prompt_embeds
        else:
            in_prompt_embeds = prompt_embeds

        # 4. Prepare timesteps
        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps, mu=1)
        elif isinstance(self.scheduler, FlowUniPCMultistepScheduler):
            self.scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
            timesteps = self.scheduler.timesteps
        elif isinstance(self.scheduler, FlowDPMSolverMultistepScheduler):
            sampling_sigmas = get_sampling_sigmas(num_inference_steps, shift)
            timesteps, _ = retrieve_timesteps(
                self.scheduler,
                device=device,
                sigmas=sampling_sigmas)
        else:
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents.
        latent_channels = self.geo_adapter.model.z_dim
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            weight_dtype,
            device,
            generator,
            latents,
        )  # [B, 16, f, 70, 140]

        # Prepare plucker embeddings
        if control_cameras is not None:  # [B, F, 6, H, W]
            # Rearrange dimensions
            # Concatenate and transpose dimensions
            control_camera_latents = control_cameras.transpose(1, 2)  # [B, 6, F, H, W]
            control_camera_latents = torch.cat(
                [torch.repeat_interleave(control_camera_latents[:, :, 0:1], repeats=4, dim=2), control_camera_latents[:, :, 1:]], dim=2
            ).transpose(1, 2).contiguous()  # [B, F+3, 6, H, W]
            control_camera_latents = control_camera_latents.view(1, (num_frames + 3) // 4, 4, control_camera_latents.shape[2], height, width).transpose(2, 3).contiguous()  # [B, (F+3)//4, 6, 4, H, W]
            # [B, 24, (F+3)//4, H, W], that is [B, 24, f, H, W]
            control_camera_latents = control_camera_latents.view(1, (num_frames + 3) // 4, control_camera_latents.shape[2] * 4, height, width).transpose(1, 2).contiguous()
            # duplicate in width dimension
            control_camera_latents = torch.cat([control_camera_latents, control_camera_latents], dim=-1)  # [B, 24, f, H, 2*W]
        else:
            control_camera_latents = None

        # Prepare control latents
        F = control_cameras.shape[1]
        B, _, C, H, W = control_images.shape
        if control_images.shape[1] == 1:
            control_images = torch.cat([control_images, torch.zeros(B, F-1, C, H, W).to(control_images.device)], dim=1)  # [B, F, 3, H, W]
            control_index = [0]
        elif control_images.shape[1] == 2:
            control_images = torch.cat([control_images[:, :1], torch.zeros(B, F-2, C, H, W).to(control_images.device), control_images[:, -1:]], dim=1)  # [B, F, 3, H, W]
            control_index = [0, -1]
        else:
            control_index = [i for i in range(F)]
        
        masks = torch.zeros(B, F, height // self.geo_adapter.spatial_compression_ratio, width // self.geo_adapter.spatial_compression_ratio * 2)
        masks[:, control_index, :, :width // self.geo_adapter.spatial_compression_ratio] = 1
        masks[:, control_index, :, width // self.geo_adapter.spatial_compression_ratio:] = 0
        masks = torch.cat([
            torch.repeat_interleave(masks[:, :1], repeats=4, dim=1), masks[:, 1:]], dim=1)  # [B, F+3, 70, 140/70]
        masks = masks.view(B, (F+3) // 4, 4, *masks.shape[-2:]).contiguous().transpose(1, 2)  # [B, 4, f, 70, 140/70]

        with torch.no_grad():
            control_image_latents = self.prepare_control_latents(
                control_images,  # in [0, 1], [B, F, 3, H, W]
                masks,  # [B, 4, f, 70, 140/70]
                weight_dtype,
                device,
            )  # [B, 20, f, 70, 140]

        # Prepare clip latent variables
        if clip_image is not None:  # in [0, 255], [H, W, 3]
            clip_image = TF.to_tensor(clip_image).sub_(0.5).div_(0.5).to(device, weight_dtype) 
            clip_context = self.clip_image_encoder([clip_image[:, None, :, :]])
        else:
            clip_context = []
            for index in control_index[:1]:
                clip_image = Image.fromarray((control_images[:, index].squeeze() * 255).permute(1, 2, 0).float().cpu().numpy().astype(np.uint8))
                clip_image = TF.to_tensor(clip_image).sub_(0.5).div_(0.5).to(device, weight_dtype) 
                clip_context.append(self.clip_image_encoder([clip_image[:, None, :, :]]))
            clip_context = torch.cat(clip_context)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        target_shape = (latent_channels, (num_frames - 1) // 4 + 1, width // self.geo_adapter.spatial_compression_ratio * 2, height // self.geo_adapter.spatial_compression_ratio)
        seq_len = math.ceil(
            (target_shape[2] * target_shape[3]) / 
            (self.transformer.config.patch_size[1] * self.transformer.config.patch_size[2]) * target_shape[1]) 
        
        # 7. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if cfg_skip_ratio is not None and i >= num_inference_steps * (1 - cfg_skip_ratio):
                    do_classifier_free_guidance = False
                    in_prompt_embeds = prompt_embeds

                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                if hasattr(self.scheduler, "scale_model_input"):
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Prepare mask latent variables
                if control_cameras is not None:
                    control_camera_latents_input = (
                        torch.cat([control_camera_latents] * 2) if do_classifier_free_guidance else control_camera_latents
                    ).to(device, weight_dtype)

                control_latents_input = (
                    torch.cat([control_image_latents] * 2) if do_classifier_free_guidance else control_image_latents
                ).to(device, weight_dtype)

                clip_context_input = (
                    torch.cat([clip_context] * 2) if do_classifier_free_guidance else clip_context
                )

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])
                
                # predict noise model_output
                with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=device):
                    noise_pred = self.transformer(
                        x=latent_model_input,
                        context=in_prompt_embeds,
                        t=timestep,
                        seq_len=seq_len,
                        y=control_latents_input,
                        y_camera=control_camera_latents_input, 
                        full_ref=None,
                        clip_fea=clip_context_input,
                    )  # [B, 16, f, 70, 140]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            results = self.decode_latents(latents, min_max_depth_mask=min_max_depth_mask)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            if output_type == "latent":
                return (latents,)
            else:
                return (
                    results['point_maps'], results['depth_maps'], results['rgbs'], 
                    results['pcds'], results['point_masks'], results['cameras']
                )

        if output_type == "latent":
            return Gen3RPipelineOutput(latents=latents)
        else:
            return Gen3RPipelineOutput(
                point_maps=results['point_maps'], depth_maps=results['depth_maps'], rgbs=results['rgbs'], 
                pcds=results['pcds'], point_masks=results['point_masks'], cameras=results['cameras']
            )
