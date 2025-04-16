# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch
from PIL import Image

from pipelines.pipeline_infu_flux import InfUFluxPipeline


def pose_synthesize(
        in_image: Image.Image,
        control_image: None | Image.Image=None,
        prompt: str="",
        base_model_path: str='black-forest-labs/FLUX.1-dev',
        model_dir: str='ByteDance/InfiniteYou',
        infu_flux_version: str='v1.0',
        model_version: str='sim_stage1',
        cuda_device: int=0,
        seed: int=0,
        width: int=512,
        height: int=512,
        guidance_scale: float=3.5,
        num_steps: int=30,
        infusenet_conditioning_scale: float=1.0,
        infusenet_guidance_start: float=0.0,
        infusenet_guidance_end: float=1.0,
        enable_realism_lora: bool=False,
        enable_anti_blur_lora: bool=False,
        quantize_8bit: bool=False,
        cpu_offload: bool=False,
):
    # Set cuda device
    torch.cuda.set_device(cuda_device)

    # Load pipeline
    infu_model_path = os.path.join(model_dir, f'infu_flux_{infu_flux_version}', model_version)
    insightface_root_path = os.path.join(model_dir, 'supports', 'insightface')
    pipe = InfUFluxPipeline(
        base_model_path=base_model_path,
        infu_model_path=infu_model_path,
        insightface_root_path=insightface_root_path,
        infu_flux_version=infu_flux_version,
        model_version=model_version,
        quantize_8bit=quantize_8bit,
        cpu_offload=cpu_offload,
    )
    # Load LoRAs (optional)
    lora_dir = os.path.join(model_dir, 'supports', 'optional_loras')
    if not os.path.exists(lora_dir): lora_dir = './models/InfiniteYou/supports/optional_loras'
    loras = []
    if enable_realism_lora:
        loras.append([os.path.join(lora_dir, 'flux_realism_lora.safetensors'), 'realism', 1.0])
    if enable_anti_blur_lora:
        loras.append([os.path.join(lora_dir, 'flux_anti_blur_lora.safetensors'), 'anti_blur', 1.0])
    pipe.load_loras(loras)
    
    # Perform inference
    if seed == 0:
        seed = torch.seed() & 0xFFFFFFFF
    image = pipe(
        id_image=in_image,
        prompt=prompt,
        control_image=control_image if control_image is not None else None,
        seed=seed,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_steps=num_steps,
        infusenet_conditioning_scale=infusenet_conditioning_scale,
        infusenet_guidance_start=infusenet_guidance_start,
        infusenet_guidance_end=infusenet_guidance_end,
        cpu_offload=cpu_offload,
    )
    return image