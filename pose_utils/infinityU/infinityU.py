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

import gc

import pillow_avif
import torch
from huggingface_hub import snapshot_download
from pillow_heif import register_heif_opener

from .pipelines.pipeline_infu_flux import InfUFluxPipeline


# Register HEIF support for Pillow
register_heif_opener()

class ModelVersion:
    STAGE_1 = "sim_stage1"
    STAGE_2 = "aes_stage2"

    DEFAULT_VERSION = STAGE_2
    
ENABLE_ANTI_BLUR_DEFAULT = False
ENABLE_REALISM_DEFAULT = False

loaded_pipeline_config = {
    "model_version": "aes_stage2",
    "enable_realism": False,
    "enable_anti_blur": False,
    'pipeline': None
}


def download_models():
    snapshot_download(repo_id='ByteDance/InfiniteYou', local_dir='./models/InfiniteYou', local_dir_use_symlinks=False)
    try:
        snapshot_download(repo_id='black-forest-labs/FLUX.1-dev', local_dir='./models/FLUX.1-dev', local_dir_use_symlinks=False)
    except Exception as e:
        print(e)
        print('\nYou are downloading `black-forest-labs/FLUX.1-dev` to `./models/FLUX.1-dev` but failed. '
              'Please accept the agreement and obtain access at https://huggingface.co/black-forest-labs/FLUX.1-dev. '
              'Then, use `huggingface-cli login` and your access tokens at https://huggingface.co/settings/tokens to authenticate. '
              'After that, run the code again.')
        print('\nYou can also download it manually from HuggingFace and put it in `./models/InfiniteYou`, '
              'or you can modify `base_model_path` in `app.py` to specify the correct path.')
        exit()


def prepare_pipeline(model_version, enable_realism, enable_anti_blur):
    if (
        loaded_pipeline_config['pipeline'] is not None
        and loaded_pipeline_config["enable_realism"] == enable_realism 
        and loaded_pipeline_config["enable_anti_blur"] == enable_anti_blur
        and model_version == loaded_pipeline_config["model_version"]
    ):
        return loaded_pipeline_config['pipeline']
    
    loaded_pipeline_config["enable_realism"] = enable_realism
    loaded_pipeline_config["enable_anti_blur"] = enable_anti_blur
    loaded_pipeline_config["model_version"] = model_version

    pipeline = loaded_pipeline_config['pipeline']
    if pipeline is None or pipeline.model_version != model_version:
        print(f'Switching model to {model_version}')
        del pipeline
        del loaded_pipeline_config['pipeline']
        gc.collect()
        torch.cuda.empty_cache()

        model_path = f'./models/InfiniteYou/infu_flux_v1.0/{model_version}'
        print(f'Loading model from {model_path}')

        pipeline = InfUFluxPipeline(
            base_model_path='./models/FLUX.1-dev',
            infu_model_path=model_path,
            insightface_root_path='./models/InfiniteYou/supports/insightface',
            image_proj_num_tokens=8,
            infu_flux_version='v1.0',
            model_version=model_version,
        )

        loaded_pipeline_config['pipeline'] = pipeline

    pipeline.pipe.delete_adapters(['realism', 'anti_blur'])
    loras = []
    if enable_realism:
        loras.append(['./models/InfiniteYou/supports/optional_loras/flux_realism_lora.safetensors', 'realism', 1.0])
    if enable_anti_blur:
        loras.append(['./models/InfiniteYou/supports/optional_loras/flux_anti_blur_lora.safetensors', 'anti_blur', 1.0])
    pipeline.load_loras(loras)

    return pipeline


def pose_synthesize(
    input_image, 
    control_image, 
    prompt, 
    seed, 
    width,
    height,
    guidance_scale, 
    num_steps, 
    infusenet_conditioning_scale, 
    infusenet_guidance_start,
    infusenet_guidance_end,
    enable_realism,
    enable_anti_blur,
    model_version
):
    pipeline = prepare_pipeline(model_version=model_version, enable_realism=enable_realism, enable_anti_blur=enable_anti_blur)

    if seed == 0:
        seed = torch.seed() & 0xFFFFFFFF

    try:
        image = pipeline(
            id_image=input_image,
            prompt=prompt,
            control_image=control_image,
            seed=seed,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            infusenet_conditioning_scale=infusenet_conditioning_scale,
            infusenet_guidance_start=infusenet_guidance_start,
            infusenet_guidance_end=infusenet_guidance_end,
        )
    except Exception as e:
        print(e)
        gr.Error(f"An error occurred: {e}")
        return gr.update()

    return gr.update(value = image, label=f"Generated Image, seed = {seed}")