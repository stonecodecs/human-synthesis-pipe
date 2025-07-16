import argparse
import os
import numpy as np

from time import time
from PIL import Image

from ic_light import BGSource, resize_without_crop, resize_and_center_crop, process_relight
from prompt_enhance import prompt_enhance
from pose_utils.infinityU.infinityU import pose_synthesize

def main():
    parser = argparse.ArgumentParser(description="IC-Light (Relighting with Foreground Condition)")
    # Pose args
    parser.add_argument("--input_img", type=str, required=True, help="Path to the input image")
    parser.add_argument("--pose_prompt", type=str, required=True, help="Prompt for the pose generation")
    parser.add_argument("--pose_seed", type=int, default=12345, help="Random seed for pose generation")
    parser.add_argument("--pose_guidance_scale", type=float, default=2.0, help="Pose guidance scale")
    parser.add_argument("--pose_steps", type=int, default=25, help="Number of steps for pose generation")
    parser.add_argument("--pose_infusenet_cond_scale", type=float, default=1.0, help="Pose InfuseNet condition scale")
    parser.add_argument("--pose_infusenet_guidance_start", type=float, default=0.5, help="Pose InfuseNet guidance start")
    parser.add_argument("--pose_infusenet_guidance_end", type=float, default=0.9, help="Pose InfuseNet guidance end")
    parser.add_argument("--pose_enable_realism", type=bool, default=True, help="Enable realism for pose generation")
    parser.add_argument("--pose_enable_anti_blur", type=bool, default=True, help="Enable anti-blur for pose generation")
    parser.add_argument("--pose_model_version", type=str, default="v1.0", help="Pose model version")
    parser.add_argument("--flux_model_path", type=str, default="/workspace/leovol/models/FLUX.1-dev", help="Path to FLUX model directory")
    parser.add_argument("--quantize_8bit", type=bool, default=True, help="Use 8-bit quantization")
    parser.add_argument("--cpu_offload", type=bool, default=True, help="Enable CPU offloading")

    # Lighting args
    parser.add_argument("--out_path", type=str, required=False, default="synthesized_images", help="Path to the output directory")
    parser.add_argument("--light_prompt", type=str, required=True, help="Prompt for the image generation")
    parser.add_argument("--light_num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--light_seed", type=int, default=12345, help="Random seed for generation")
    parser.add_argument("--light_steps", type=int, default=25, help="Number of steps for generation")
    parser.add_argument("--light_a_prompt", type=str, default="best quality", help="Added prompt")
    parser.add_argument("--light_n_prompt", type=str, default="lowres, bad anatomy, bad hands, cropped, worst quality", help="Negative prompt")
    parser.add_argument("--light_cfg", type=float, default=2.0, help="CFG scale")
    parser.add_argument("--light_highres_scale", type=float, default=1.5, help="Highres scale")
    parser.add_argument("--light_highres_denoise", type=float, default=0.5, help="Highres denoise")
    parser.add_argument("--light_lowres_denoise", type=float, default=0.9, help="Lowres denoise")
    parser.add_argument("--light_bg_source", type=str, default=BGSource.NONE.value, help="Background source")
    args = parser.parse_args()

    # Create output directory
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    # Read and extract input image information
    input_image = Image.open(args.input_img).convert("RGB")
    h, w = input_image.size

    # Enhance pose prompt
    pose_prompt = prompt_enhance(input_image, args.pose_prompt)

    # Generate pose
    # Create black control image
    control_image = None
    pose_image = pose_synthesize(
        input_image,
        control_image,
        pose_prompt,
        args.flux_model_path,  # Use command line parameter
        'ByteDance/InfiniteYou',
        'v1.0',
        args.pose_model_version,
        0,
        args.pose_seed,
        w,
        h,
        args.pose_guidance_scale,
        args.pose_steps,
        args.pose_infusenet_cond_scale,
        args.pose_infusenet_guidance_start,
        args.pose_infusenet_guidance_end,
        args.pose_enable_realism,
        args.pose_enable_anti_blur,
        True,
        True
    )
    
    # Enhance prompt
    light_prompt = prompt_enhance(pose_image, args.light_prompt)

    input_fg, results = process_relight(
        pose_image,
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
    Image.fromarray(results[0]).save(os.path.join(args.out_path, "output.png"))

if __name__ == "__main__":
    main()
