import argparse
import os
import numpy as np

from time import time
from PIL import Image

from ic_light import BGSource, resize_without_crop, resize_and_center_crop, process_relight
from prompting.prompt_enhance import prompt_enhance_pose
from pose_utils.infinityU.infinityU import pose_synthesize

def main():
    parser = argparse.ArgumentParser(description="IC-Light (Relighting with Foreground Condition)")
    # Pose args
    parser.add_argument("--input_img", type=str, required=True, help="Path to the input image")
    parser.add_argument("--pose_prompt", type=str, required=True, help="Prompt for the pose generation")
    parser.add_argument("--pose_seed", type=int, default=12345, help="Random seed for pose generation")
    parser.add_argument("--pose_guidance_scale", type=float, default=2.0, help="Pose guidance scale")
    parser.add_argument("--pose_steps", type=int, default=25, help="Number of steps for pose generation")
    parser.add_argument("--flux_model_path", type=str, default="/workspace/stonevol/ext_models/FLUX.1-dev", help="Path to FLUX model directory")
    parser.add_argument("--infinite_model_path", type=str, default="ByteDance/InfiniteYou", help="Path to InfiniteYou model directory")
    parser.add_argument("--pose_infusenet_cond_scale", type=float, default=1.0, help="Pose InfuseNet condition scale")
    parser.add_argument("--pose_infusenet_guidance_start", type=float, default=0.5, help="Pose InfuseNet guidance start")
    parser.add_argument("--pose_infusenet_guidance_end", type=float, default=0.9, help="Pose InfuseNet guidance end")
    parser.add_argument("--pose_enable_realism", type=bool, default=True, help="Enable realism for pose generation")
    parser.add_argument("--pose_enable_anti_blur", type=bool, default=True, help="Enable anti-blur for pose generation")
    parser.add_argument("--pose_model_version", type=str, default="v1.0", help="Pose model version")
    parser.add_argument("--quantize_8bit", type=bool, default=True, help="Use 8-bit quantization")
    parser.add_argument("--cpu_offload", type=bool, default=True, help="Enable CPU offloading")
    parser.add_argument("--out_path", type=str, required=False, default="synthesized_images", help="Path to the output directory")
    args = parser.parse_args()

    # Create output directory
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    # Read and extract input image information
    input_image = Image.open(args.input_img).convert("RGB")
    h, w = input_image.size
    h = (h - h %8)
    w = (w - w%8)

    # Enhance pose prompt
    # pose_prompt = prompt_enhance_pose(input_image, args.pose_prompt)
    pose_prompt = args.pose_prompt

    # Generate pose
    # Create black control image
    control_image = None
    pose_image = pose_synthesize(
        input_image,
        control_image,
        pose_prompt,
        args.flux_model_path,
        args.infinite_model_path,
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
        False,
        True,
        True
    )
    
    pose_image.save(os.path.join(args.out_path, "pose.png"))

if __name__ == "__main__":
    main()
