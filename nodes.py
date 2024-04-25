import argparse
import datetime
import logging
import inspect
import math
import os

import folder_paths
comfy_path = os.path.dirname(folder_paths.__file__)

import sys
SimDA_path=f'{comfy_path}/custom_nodes/ComfyUI-SimDA'
sys.path.insert(0,SimDA_path)

from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from simda.models.unet import UNet3DConditionModel
from simda.data.dataset import SimDADataset
from simda.pipelines.pipeline_simda import SimDAPipeline
from simda.util import save_videos_grid, ddim_inversion
from einops import rearrange

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")

import json

import comfy.utils

def train(
    pretrained_model_path: str,
    output_dir: str,
    train_data: Dict,
    validation_data: Dict,
    validation_steps: int = 100,
    trainable_modules: Tuple[str] = (
        "temp_adapter"
        "adapter_s"
        "adapter_ffn"
    ),
    train_batch_size: int = 1,
    max_train_steps: int = 500,
    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = True,
    checkpointing_steps: int = 500,
    resume_from_checkpoint: Optional[str] = None,
    mixed_precision: Optional[str] = "fp16",
    use_8bit_adam: bool = False,
    enable_xformers_memory_efficient_attention: bool = True,
    seed: Optional[int] = None,
):
    with torch.inference_mode(False):
        with torch.set_grad_enabled(True):
            *_, config = inspect.getargvalues(inspect.currentframe())

            accelerator = Accelerator(
                gradient_accumulation_steps=gradient_accumulation_steps,
                mixed_precision=mixed_precision,
            )
            print(f'is_main_process{accelerator.is_main_process}')
            print(f'num_processes{accelerator.num_processes}')

            # Make one log on every process with the configuration for debugging.
            logging.basicConfig(
                format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                datefmt="%m/%d/%Y %H:%M:%S",
                level=logging.INFO,
            )
            logger.info(accelerator.state, main_process_only=False)
            if accelerator.is_local_main_process:
                transformers.utils.logging.set_verbosity_warning()
                diffusers.utils.logging.set_verbosity_info()
            else:
                transformers.utils.logging.set_verbosity_error()
                diffusers.utils.logging.set_verbosity_error()

            # If passed along, set the training seed now.
            if seed is not None:
                set_seed(seed)

            # Handle the output folder creation
            if accelerator.is_main_process:
                # now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
                # output_dir = os.path.join(output_dir, now)
                os.makedirs(output_dir, exist_ok=True)
                os.makedirs(f"{output_dir}/samples", exist_ok=True)
                os.makedirs(f"{output_dir}/inv_latents", exist_ok=True)
                OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

            # Load scheduler, tokenizer and models.
            noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
            tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
            text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
            vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
            unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet")

            # Freeze vae and text_encoder
            vae.requires_grad_(False)
            text_encoder.requires_grad_(False)

            unet.requires_grad_(False)
            for name, module in unet.named_modules():
                if name.endswith(tuple(trainable_modules)):
                    for params in module.parameters():
                        params.requires_grad = True

            if enable_xformers_memory_efficient_attention:
                if is_xformers_available():
                    unet.enable_xformers_memory_efficient_attention()
                else:
                    raise ValueError("xformers is not available. Make sure it is installed correctly")

            if gradient_checkpointing:
                unet.enable_gradient_checkpointing()

            if scale_lr:
                learning_rate = (
                    learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
                )

            # Initialize the optimizer
            if use_8bit_adam:
                try:
                    import bitsandbytes as bnb
                except ImportError:
                    raise ImportError(
                        "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
                    )

                optimizer_cls = bnb.optim.AdamW8bit
            else:
                optimizer_cls = torch.optim.AdamW

            optimizer = optimizer_cls(
                unet.parameters(),
                lr=learning_rate,
                betas=(adam_beta1, adam_beta2),
                weight_decay=adam_weight_decay,
                eps=adam_epsilon,
            )

            # Get the training dataset
            train_dataset = SimDADataset(**train_data)

            # Preprocessing the dataset
            train_dataset.prompt_ids = tokenizer(
                train_dataset.prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids[0]

            # DataLoaders creation:
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=train_batch_size
            )

            # Get the validation pipeline
            validation_pipeline = SimDAPipeline(
                vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
                scheduler=DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
            )
            validation_pipeline.enable_vae_slicing()
            ddim_inv_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler')
            ddim_inv_scheduler.set_timesteps(validation_data.num_inv_steps)

            # Scheduler
            lr_scheduler = get_scheduler(
                lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
                num_training_steps=max_train_steps * gradient_accumulation_steps,
            )

            # Prepare everything with our `accelerator`.
            unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, optimizer, train_dataloader, lr_scheduler
            )

            # For mixed precision training we cast the text_encoder and vae weights to half-precision
            # as these models are only used for inference, keeping weights in full precision is not required.
            weight_dtype = torch.float32
            if accelerator.mixed_precision == "fp16":
                weight_dtype = torch.float16
            elif accelerator.mixed_precision == "bf16":
                weight_dtype = torch.bfloat16

            # Move text_encode and vae to gpu and cast to weight_dtype
            text_encoder.to(accelerator.device, dtype=weight_dtype)
            vae.to(accelerator.device, dtype=weight_dtype)

            # We need to recalculate our total training steps as the size of the training dataloader may have changed.
            num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
            # Afterwards we recalculate our number of training epochs
            num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

            # We need to initialize the trackers we use, and also store our configuration.
            # The trackers initializes automatically on the main process.
            if accelerator.is_main_process:
                accelerator.init_trackers("text2video-fine-tune")

            # Train!
            total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {len(train_dataset)}")
            logger.info(f"  Num Epochs = {num_train_epochs}")
            logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
            logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
            logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
            logger.info(f"  Total optimization steps = {max_train_steps}")
            global_step = 0
            first_epoch = 0

            # Potentially load in the weights and states from a previous save
            if resume_from_checkpoint:
                if resume_from_checkpoint != "latest":
                    path = os.path.basename(resume_from_checkpoint)
                else:
                    # Get the most recent checkpoint
                    dirs = os.listdir(output_dir)
                    dirs = [d for d in dirs if d.startswith("checkpoint")]
                    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                    path = dirs[-1]
                accelerator.print(f"Resuming from checkpoint {path}")
                accelerator.load_state(os.path.join(output_dir, path))
                global_step = int(path.split("-")[1])

                first_epoch = global_step // num_update_steps_per_epoch
                resume_step = global_step % num_update_steps_per_epoch

            # Only show the progress bar once on each machine.
            progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
            progress_bar.set_description("Steps")
            pbar = comfy.utils.ProgressBar(max_train_steps)

            for epoch in range(first_epoch, num_train_epochs):
                unet.train()
                train_loss = 0.0
                for step, batch in enumerate(train_dataloader):
                    # Skip steps until we reach the resumed step
                    if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                        if step % gradient_accumulation_steps == 0:
                            progress_bar.update(1)
                            pbar.update(1)
                        continue

                    with accelerator.accumulate(unet):
                        # Convert videos to latent space
                        pixel_values = batch["pixel_values"].to(weight_dtype)
                        video_length = pixel_values.shape[1]
                        pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                        latents = vae.encode(pixel_values).latent_dist.sample()
                        latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                        latents = latents * 0.18215

                        # Sample noise that we'll add to the latents
                        noise = torch.randn_like(latents)
                        bsz = latents.shape[0]
                        # Sample a random timestep for each video
                        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                        timesteps = timesteps.long()

                        # Add noise to the latents according to the noise magnitude at each timestep
                        # (this is the forward diffusion process)
                        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                        # Get the text embedding for conditioning
                        encoder_hidden_states = text_encoder(batch["prompt_ids"])[0]

                        # Get the target for loss depending on the prediction type
                        if noise_scheduler.prediction_type == "epsilon":
                            target = noise
                        elif noise_scheduler.prediction_type == "v_prediction":
                            target = noise_scheduler.get_velocity(latents, noise, timesteps)
                        else:
                            raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")

                        # Predict the noise residual and compute loss
                        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                        # Gather the losses across all processes for logging (if we use distributed training).
                        avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                        train_loss += avg_loss.item() / gradient_accumulation_steps

                        # Backpropagate
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        progress_bar.update(1)
                        pbar.update(1)
                        global_step += 1
                        accelerator.log({"train_loss": train_loss}, step=global_step)
                        train_loss = 0.0

                        if global_step % checkpointing_steps == 0:
                            if accelerator.is_main_process:
                                save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                                accelerator.save_state(save_path)
                                logger.info(f"Saved state to {save_path}")

                        if global_step % validation_steps == 0:
                            if accelerator.is_main_process:
                                samples = []
                                generator = torch.Generator(device=latents.device)
                                generator.manual_seed(seed)

                                ddim_inv_latent = None
                                if validation_data.use_inv_latent:
                                    inv_latents_path = os.path.join(output_dir, f"inv_latents/ddim_latent-{global_step}.pt")
                                    ddim_inv_latent = ddim_inversion(
                                        validation_pipeline, ddim_inv_scheduler, video_latent=latents,
                                        num_inv_steps=validation_data.num_inv_steps, prompt="")[-1].to(weight_dtype)
                                    torch.save(ddim_inv_latent, inv_latents_path)

                                for idx, prompt in enumerate(validation_data.prompts):
                                    sample = validation_pipeline(prompt, generator=generator, latents=ddim_inv_latent,
                                                                **validation_data).videos
                                    save_videos_grid(sample, f"{output_dir}/samples/sample-{global_step}/{prompt}.gif")
                                    samples.append(sample)
                                samples = torch.concat(samples)
                                save_path = f"{output_dir}/samples/sample-{global_step}.gif"
                                save_videos_grid(samples, save_path)
                                logger.info(f"Saved samples to {save_path}")

                    logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                    progress_bar.set_postfix(**logs)

                    if global_step >= max_train_steps:
                        break

            # Create the pipeline using the trained modules and save it.
            accelerator.wait_for_everyone()
            #if accelerator.is_main_process:
            unet = accelerator.unwrap_model(unet)
            pipeline = SimDAPipeline.from_pretrained(
                pretrained_model_path,
                text_encoder=text_encoder,
                vae=vae,
                unet=unet,
            )
            pipeline.save_pretrained(output_dir)

            accelerator.end_training()

class SimDATrain:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pretrained_model_path": (os.listdir(os.path.join(folder_paths.models_dir,"diffusers")), {"default": "stable-diffusion-v1-5"}),
                "output_dir": ("STRING", {"default": "car-turn"}),
                "video_path": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": "a jeep car is moving on the road"}),
                "n_sample_frames":("INT",{"default":24}),
                "width":("INT",{"default":512}),
                "height":("INT",{"default":512}),
                "sample_start_idx":("INT",{"default":16}),
                "sample_frame_rate":("INT",{"default":2}),
                "validation_data_prompts":("STRING",{"multiline": True, "default":'["a jeep car is moving on the beach"]'}),
                "num_inference_steps":("INT",{"default":20}),
                "guidance_scale":("FLOAT",{"default":12.5}),
                "max_train_steps":("INT",{"default":300}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "CameraCtrl"

    def run(self,pretrained_model_path,output_dir,video_path,prompt,n_sample_frames,width,height,sample_start_idx,sample_frame_rate,validation_data_prompts,num_inference_steps,guidance_scale,max_train_steps):
        pretrained_model_path=os.path.join(os.path.join(folder_paths.models_dir,"diffusers"),pretrained_model_path)
        output_dir=os.path.join(folder_paths.output_directory,output_dir)

        config = OmegaConf.load(os.path.join(SimDA_path,"configs/car-turn.yaml"))
        OmegaConf.update(config, "pretrained_model_path", pretrained_model_path)
        OmegaConf.update(config, "output_dir", output_dir)

        OmegaConf.update(config, "train_data.video_path", video_path)
        OmegaConf.update(config, "train_data.prompt", prompt)
        OmegaConf.update(config, "train_data.n_sample_frames", n_sample_frames)
        OmegaConf.update(config, "train_data.width", width)
        OmegaConf.update(config, "train_data.height", height)
        OmegaConf.update(config, "train_data.sample_start_idx", sample_start_idx)
        OmegaConf.update(config, "train_data.sample_frame_rate", sample_frame_rate)

        OmegaConf.update(config, "validation_data.prompts", json.loads(validation_data_prompts))
        OmegaConf.update(config, "validation_data.num_inference_steps", num_inference_steps)
        OmegaConf.update(config, "validation_data.guidance_scale", guidance_scale)

        OmegaConf.update(config, "max_train_steps", max_train_steps)

        train(**config)

        return (output_dir,)

class SimDALoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pretrained_model_path": (os.listdir(os.path.join(folder_paths.models_dir,"diffusers")), {"default": "stable-diffusion-v1-5"}),
                "my_model_path": ("STRING", {"default": ""}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("SimDAPipe",)
    FUNCTION = "run"
    CATEGORY = "CameraCtrl"

    def run(self,pretrained_model_path,my_model_path):
        pretrained_model_path=os.path.join(os.path.join(folder_paths.models_dir,"diffusers"),pretrained_model_path)
        #my_model_path = "./outputs/car-turn"
        unet = UNet3DConditionModel.from_pretrained(my_model_path, subfolder='unet', torch_dtype=torch.float16).to('cuda')
        pipe = SimDAPipeline.from_pretrained(pretrained_model_path, unet=unet, torch_dtype=torch.float16).to("cuda")
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_vae_slicing()

        return (pipe,)


class SimDARun:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("SimDAPipe",),
                "prompt": ("STRING", {"default": "a jeep car is moving on the road"}),
                "video_length":("INT",{"default":24}),
                "width":("INT",{"default":512}),
                "height":("INT",{"default":512}),
                "num_inference_steps":("INT",{"default":50}),
                "guidance_scale":("FLOAT",{"default":12.5}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "CameraCtrl"

    def run(self,pipe,prompt,video_length,width,height,num_inference_steps,guidance_scale):
        video = pipe(prompt, latents=None, video_length=video_length, height=height, width=width, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).videos

        video = rearrange(video, "b c t h w -> b t h w c")
        return video

class VHS_FILENAMES_STRING_SimDA:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "filenames": ("VHS_FILENAMES",),
                    }
                }

    RETURN_TYPES = ("STRING",)
    CATEGORY = "StreamingT2V"
    FUNCTION = "run"

    def run(self, filenames):
        return (filenames[1][-1],)
    
NODE_CLASS_MAPPINGS = {
    "SimDATrain":SimDATrain,
    "SimDALoader":SimDALoader,
    "SimDARun":SimDARun,
    "VHS_FILENAMES_STRING_SimDA":VHS_FILENAMES_STRING_SimDA,
}