import os
import math
import numpy as np
import torch
import safetensors.torch as sf
import argparse
import random

from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from briarmbg import BriaRMBG
from enum import Enum
from torch.hub import download_url_to_file
from tqdm import tqdm
from time import time
from prompt_enhance import prompt_enhance_light
from crop import create_transforms_json, crop_image, apply_mask
from prompt_sampler import PromptSampler
from tqdm import tqdm
import json
import cv2

# 'stablediffusionapi/realistic-vision-v51'
# 'runwayml/stable-diffusion-v1-5'
sd15_name = 'stablediffusionapi/realistic-vision-v51'
tokenizer = CLIPTokenizer.from_pretrained(sd15_name, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(sd15_name, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet")
rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4")

# Change UNet

with torch.no_grad():
    new_conv_in = torch.nn.Conv2d(8, unet.conv_in.out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding)
    new_conv_in.weight.zero_()
    new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
    new_conv_in.bias = unet.conv_in.bias
    unet.conv_in = new_conv_in

unet_original_forward = unet.forward


def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
    c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
    c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
    new_sample = torch.cat([sample, c_concat], dim=1)
    kwargs['cross_attention_kwargs'] = {}
    return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)


unet.forward = hooked_unet_forward

# Load

model_path = './models/iclight_sd15_fc.safetensors'

if not os.path.exists(model_path):
    download_url_to_file(url='https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors', dst=model_path)

sd_offset = sf.load_file(model_path)
sd_origin = unet.state_dict()
keys = sd_origin.keys()
sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
unet.load_state_dict(sd_merged, strict=True)
del sd_offset, sd_origin, sd_merged, keys

# Device

device = torch.device('cuda')
text_encoder = text_encoder.to(device=device, dtype=torch.float16)
vae = vae.to(device=device, dtype=torch.bfloat16)
unet = unet.to(device=device, dtype=torch.float16)
rmbg = rmbg.to(device=device, dtype=torch.float32)

# SDP

unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

# Samplers

ddim_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

euler_a_scheduler = EulerAncestralDiscreteScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    steps_offset=1
)

dpmpp_2m_sde_karras_scheduler = DPMSolverMultistepScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    algorithm_type="sde-dpmsolver++",
    use_karras_sigmas=True,
    steps_offset=1
)

# Pipelines

t2i_pipe = StableDiffusionPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
)

i2i_pipe = StableDiffusionImg2ImgPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
)


@torch.inference_mode()
def encode_prompt_inner(txt: str):
    max_length = tokenizer.model_max_length
    chunk_length = tokenizer.model_max_length - 2
    id_start = tokenizer.bos_token_id
    id_end = tokenizer.eos_token_id
    id_pad = id_end

    def pad(x, p, i):
        return x[:i] if len(x) >= i else x + [p] * (i - len(x))

    tokens = tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
    chunks = [[id_start] + tokens[i: i + chunk_length] + [id_end] for i in range(0, len(tokens), chunk_length)]
    chunks = [pad(ck, id_pad, max_length) for ck in chunks]

    token_ids = torch.tensor(chunks).to(device=device, dtype=torch.int64)
    conds = text_encoder(token_ids).last_hidden_state

    return conds


@torch.inference_mode()
def encode_prompt_pair(positive_prompt, negative_prompt):
    c = encode_prompt_inner(positive_prompt)
    uc = encode_prompt_inner(negative_prompt)

    c_len = float(len(c))
    uc_len = float(len(uc))
    max_count = max(c_len, uc_len)
    c_repeat = int(math.ceil(max_count / c_len))
    uc_repeat = int(math.ceil(max_count / uc_len))
    max_chunk = max(len(c), len(uc))

    c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
    uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

    c = torch.cat([p[None, ...] for p in c], dim=1)
    uc = torch.cat([p[None, ...] for p in uc], dim=1)

    return c, uc


@torch.inference_mode()
def pytorch2numpy(imgs, quant=True):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)

        if quant:
            y = y * 127.5 + 127.5
            y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        else:
            y = y * 0.5 + 0.5
            y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)

        results.append(y)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0  # so that 127 must be strictly 0.0
    h = h.movedim(-1, 1)
    return h


def resize_and_center_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.array(cropped_image)


def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)


@torch.inference_mode()
def run_rmbg(img, sigma=0.0):
    H, W, C = img.shape
    assert C == 3
    k = (256.0 / float(H * W)) ** 0.5
    feed = resize_without_crop(img, int(64 * round(W * k)), int(64 * round(H * k)))
    feed = numpy2pytorch([feed]).to(device=device, dtype=torch.float32)
    alpha = rmbg(feed)[0][0]
    alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
    alpha = alpha.movedim(1, -1)[0]
    alpha = alpha.detach().float().cpu().numpy().clip(0, 1)
    result = 127 + (img.astype(np.float32) - 127 + sigma) * alpha
    return result.clip(0, 255).astype(np.uint8), alpha


@torch.inference_mode()
def process(input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source):
    bg_source = BGSource(bg_source)
    input_bg = None

    if bg_source == BGSource.NONE:
        pass
    elif bg_source == BGSource.LEFT:
        gradient = np.linspace(255, 0, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.RIGHT:
        gradient = np.linspace(0, 255, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.TOP:
        gradient = np.linspace(255, 0, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.BOTTOM:
        gradient = np.linspace(0, 255, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    else:
        raise 'Wrong initial latent!'

    rng = torch.Generator(device=device).manual_seed(int(seed))

    fg = resize_and_center_crop(input_fg, image_width, image_height)

    concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor

    conds, unconds = encode_prompt_pair(positive_prompt=prompt + ', ' + a_prompt, negative_prompt=n_prompt)

    if input_bg is None:
        latents = t2i_pipe(
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=image_width,
            height=image_height,
            num_inference_steps=steps,
            num_images_per_prompt=num_samples,
            generator=rng,
            output_type='latent',
            guidance_scale=cfg,
            cross_attention_kwargs={'concat_conds': concat_conds},
        ).images.to(vae.dtype) / vae.config.scaling_factor
    else:
        bg = resize_and_center_crop(input_bg, image_width, image_height)
        bg_latent = numpy2pytorch([bg]).to(device=vae.device, dtype=vae.dtype)
        bg_latent = vae.encode(bg_latent).latent_dist.mode() * vae.config.scaling_factor
        latents = i2i_pipe(
            image=bg_latent,
            strength=lowres_denoise,
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=image_width,
            height=image_height,
            num_inference_steps=int(round(steps / lowres_denoise)),
            num_images_per_prompt=num_samples,
            generator=rng,
            output_type='latent',
            guidance_scale=cfg,
            cross_attention_kwargs={'concat_conds': concat_conds},
        ).images.to(vae.dtype) / vae.config.scaling_factor

    pixels = vae.decode(latents).sample
    pixels = pytorch2numpy(pixels)
    pixels = [resize_without_crop(
        image=p,
        target_width=int(round(image_width * highres_scale / 64.0) * 64),
        target_height=int(round(image_height * highres_scale / 64.0) * 64))
    for p in pixels]

    pixels = numpy2pytorch(pixels).to(device=vae.device, dtype=vae.dtype)
    latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
    latents = latents.to(device=unet.device, dtype=unet.dtype)

    image_height, image_width = latents.shape[2] * 8, latents.shape[3] * 8

    fg = resize_and_center_crop(input_fg, image_width, image_height)
    concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor

    latents = i2i_pipe(
        image=latents,
        strength=highres_denoise,
        prompt_embeds=conds,
        negative_prompt_embeds=unconds,
        width=image_width,
        height=image_height,
        num_inference_steps=int(round(steps / highres_denoise)),
        num_images_per_prompt=num_samples,
        generator=rng,
        output_type='latent',
        guidance_scale=cfg,
        cross_attention_kwargs={'concat_conds': concat_conds},
    ).images.to(vae.dtype) / vae.config.scaling_factor

    pixels = vae.decode(latents).sample

    return pytorch2numpy(pixels)


@torch.inference_mode()
def process_relight(input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source):
    input_fg, matting = run_rmbg(input_fg)
    results = process(input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source)
    return input_fg, results

def enhance_prompt_from_json(subject, prompt_json, prompt):
    base_prompt = prompt_json[subject] + ", " + prompt
    return base_prompt

quick_prompts = [
    'sunshine from window',
    'neon light, city', 
    'sunset over sea',
    'golden time',
    'sci-fi RGB glowing, cyberpunk',
    'natural lighting',
    'warm atmosphere, at home, bedroom',
    'magic lit',
    'evil, gothic, Yharnam',
    'light and shadow',
    'shadow from window',
    'soft studio lighting',
    'home atmosphere, cozy bedroom illumination',
    'neon, Wong Kar-wai, warm',
    'Soft natural light streaming through a large north-facing window on an overcast day',
    'Warm afternoon sunlight filtering through sheer curtains in a living room',
    'Harsh fluorescent ceiling lights in a typical office cubicle environment',
    'Golden hour sunlight hitting the subject from a low angle during late afternoon',
    "Cool blue light from a computer screen illuminating someone's face in a dark room",
    'Bright white LED desk lamp providing focused task lighting for reading or work',
    'Warm yellow light from a bedside table lamp creating a cozy nighttime atmosphere',
    'Natural daylight from sliding glass doors leading to a backyard patio',
    'Kitchen pendant lights hanging over an island providing focused downward illumination',
    'Bathroom vanity mirror lights creating even front-facing illumination for grooming',
    'Car headlights illuminating a person standing in a parking lot at night',
    'Street lamp casting a pool of orange sodium light on a sidewalk',
    'Living room floor lamp with a fabric shade creating soft ambient lighting',
    'Natural sunlight bouncing off white walls in a bright, airy bedroom',
    'Overhead track lighting in a retail store creating multiple directional light sources',
    'Campfire light flickering and casting warm orange tones on faces around the fire',
    'Television screen glow providing the only light source in a darkened family room',
    'Morning sunlight streaming through venetian blinds creating striped shadow patterns',
    'Porch light fixture illuminating the front entrance of a house at dusk',
    'Restaurant booth lighting with warm pendant lights creating intimate dining atmosphere'
]
quick_prompts = [[x] for x in quick_prompts]


quick_subjects = [
    'beautiful woman, detailed face',
    'handsome man, detailed face',
]
quick_subjects = [[x] for x in quick_subjects]


class BGSource(Enum):
    NONE = "None"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"

def get_pod_index():
    """Get the pod index from environment variables."""
    # Try different ways to get pod index
    return int(os.environ.get('JOB_COMPLETION_INDEX', 0))

def get_assigned_subjects(root_dir, pod_index, total_pods):
    """Get the list of subjects assigned to this pod based on index."""
    all_subjects = []
    for item in os.listdir(root_dir):
        subject_path = os.path.join(root_dir, item)
        if os.path.isdir(subject_path):
            all_subjects.append(item)
    
    # Sort for consistent partitioning
    all_subjects = sorted(all_subjects)
    
    # Assign subjects to pods using modulo
    assigned_subjects = []
    for i, subject in enumerate(all_subjects):
        if i % total_pods == pod_index:
            assigned_subjects.append(subject)
    
    return assigned_subjects

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ICLight")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory")
    parser.add_argument("--out_path", type=str, required=False, default="relit_images", help="Path to the output directory")
    # parser.add_argument("--light_prompt", type=str, required=True, help="Prompt for the image generation")
    parser.add_argument("--light_num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--light_seed", type=int, default=12345, help="Random seed for generation")
    parser.add_argument("--light_steps", type=int, default=25, help="Number of steps for generation")
    parser.add_argument("--light_a_prompt", type=str, default="best quality", help="Added prompt")
    parser.add_argument("--light_n_prompt", type=str, default="lowres, bad anatomy, bad hands, cropped, worst quality", help="Negative prompt")
    parser.add_argument("--light_cfg", type=float, default=2.0, help="CFG scale")
    parser.add_argument("--light_highres_scale", type=float, default=1024/576, help="Highres scale")
    parser.add_argument("--light_highres_denoise", type=float, default=0.5, help="Highres denoise")
    parser.add_argument("--light_lowres_denoise", type=float, default=0.9, help="Lowres denoise")
    parser.add_argument("--light_bg_source", type=str, default=BGSource.NONE.value, help="Background source")
    parser.add_argument("--step_size", type=int, default=20, help="Step size (of timesteps) to relight")
    parser.add_argument("--caption_json", type=str, default="/workspace/datasetvol/mvhuman_data/text_description_48.json", help="Path to the caption json file")
    parser.add_argument("--prompt_file", type=str, default="/workspace/datasetvol/light_prompts.txt", help="Path to the prompt file")
    args = parser.parse_args()

    pod_index = get_pod_index()
    total_pods = int(os.environ.get('JOB_PARALLELISM', 1))
    pod_id = os.environ.get('HOSTNAME', f'pod_{random.randint(1000, 9999)}')
    assigned_subjects = get_assigned_subjects(args.input_dir, pod_index, total_pods)
    print(f"Pod {pod_id} assigned subjects {assigned_subjects[:5]}...")

    # Create output directory with structure the same as the MVHN dataset dir
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    # load the prompt json file
    with open(args.caption_json, 'r') as f:
        caption_json = json.load(f)

    prompt_sampler = PromptSampler(args.prompt_file, num_samples=1)
    paths_with_errors = []

    # for each subject
    for subject in tqdm(assigned_subjects, desc="Processing Subjects"):
        # for each camera
        camera_ids = sorted(os.listdir(os.path.join(args.input_dir, subject, 'images_lr')))
        for camera_id in tqdm(camera_ids, desc=f"Processing Cameras for Subject {subject}", leave=False):
            # for each time step
            timestep_dir = os.path.join(args.input_dir, subject, 'images_lr', camera_id)
            time_steps = sorted(os.listdir(timestep_dir))
            for time_step in tqdm(time_steps[::args.step_size], desc=f"Processing Time Steps for Camera {camera_id}", leave=False):
                timestep = time_step[:4]
                if os.path.exists(os.path.join(args.out_path, subject, 'images_lr', camera_id, f"{timestep}_img.png")):
                    continue # if already processed, skip
                image_path = os.path.join(timestep_dir, f"{timestep}_img.jpg")
                mask_path = os.path.join(timestep_dir, f"{timestep}_img_fmask.png").replace('images_lr', 'fmask_lr')

                # crop image (based on annots center)
                try:
                    cropped_image = crop_image(
                        image_path,
                        mask_path
                    )
               

                    np_cropped = np.array(cropped_image)
                    np_cropped = cv2.resize(np_cropped, (576, 576), interpolation=cv2.INTER_LANCZOS4)

                    # enhance prompt
                    prompt = prompt_sampler.sample_prompt()[0]
                    light_prompt = enhance_prompt_from_json(subject, caption_json, prompt)
                    print(f"Prompt: {light_prompt}")
                
                    # relight the image
                    h, w, _ = np_cropped.shape

                    h = h - (h % 8)
                    w = w - (w % 8)

                    input_fg, results = process_relight(
                        np_cropped,
                        light_prompt,
                        w,
                        h,
                        args.light_num_samples,
                        args.light_seed,
                        args.light_steps,
                        args.light_a_prompt,
                        args.light_n_prompt,
                        args.light_cfg,
                        args.light_highres_scale,
                        args.light_highres_denoise,
                        args.light_lowres_denoise,
                        BGSource(args.light_bg_source)
                    )

                    # Assume that we just generate one image for simplicity
                    # Save the output image
                    os.makedirs(os.path.join(args.out_path, subject, 'images_lr', camera_id), exist_ok=True)
                    Image.fromarray(results[0]).save(os.path.join(args.out_path, subject, 'images_lr', camera_id, f"{timestep}_img.png"))
                    print(f"Saved image to {os.path.join(args.out_path, subject, 'images_lr', camera_id, f'{timestep}_img.png')}")

                except Exception as e: # if any error occurs, skip this image and store an error log
                    print(f"Error cropping image {image_path}: {e}")
                    paths_with_errors.append(image_path)
                    continue

    print(f"[ICLight] Pod {pod_id} completed generations!")
    print(f"[ICLight] Pod {pod_id} had {len(paths_with_errors)} errors.")

    # Write paths with errors to file
    error_log_path = os.path.join(args.out_path, f'iclight_errors.txt')
    with open(error_log_path, 'a') as f:
        for path in paths_with_errors:
            f.write(f"{path}\n")
    print(f"Wrote {len(paths_with_errors)} error paths to {error_log_path}")

    # for timestep in tqdm(all_time_steps, desc="Processing Time Steps"):
    #     # Relight
    #     for camera_id in tqdm(camera_ids, desc=f"Processing Cameras for Timestep {timestep}", leave=False):
    #         # Get image
    #         image_path = os.path.join(args.input_dir, 'images_lr', camera_id, f"{timestep}_img.jpg")
    #         pose_image = Image.open(image_path)
    #         pose_image = np.array(pose_image)
    #         mask_path = os.path.join(args.input_dir, 'fmask_lr', camera_id, f"{timestep}_img_fmask.png")
    #         mask_image = np.array(Image.open(mask_path))
    #         img_masked = apply_mask(pose_image, mask_image)
    #         img_masked_pil = Image.fromarray(img_masked)
    #         crop = crop_params[camera_id]['crop']
    #         cropped = img_masked_pil.crop(crop)
    #         np_cropped = np.array(cropped)
        
    #         # Enhance prompt
    #         light_prompt = prompt_enhance_light(cropped, args.light_prompt)
        
    #         h, w, _ = np_cropped.shape

    #         h = h - (h % 8)
    #         w = w - (w % 8)

    #         input_fg, results = process_relight(
    #             np_cropped,
    #             light_prompt,
    #             w,
    #             h,
    #             args.light_num_samples,
    #             args.light_seed,
    #             args.light_steps,
    #             args.light_a_prompt,
    #             args.light_n_prompt,
    #             args.light_cfg,
    #             args.light_highres_scale,
    #             args.light_highres_denoise,
    #             args.light_lowres_denoise,
    #             BGSource(args.light_bg_source)
    #         )

    #         # Assume that we just generate one image for simplicity
    #         # Save the output image
    #         os.makedirs(os.path.join(args.out_path, 'images_lr', camera_id), exist_ok=True)
    #         Image.fromarray(results[0]).save(os.path.join(args.out_path, 'images_lr', camera_id, f"{timestep}_image.png"))