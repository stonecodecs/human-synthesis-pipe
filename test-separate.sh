#/bin/bash

export INPUT_IMAGE="/workspace/datasetvol/mvhuman_data/mv_captures/100001/images_lr/CC32871A023/0010_img.jpg"
export POSE_PROMPT="a woman crossing her arms"
export POSE_SEED=12345
export POSE_GUIDANCE_SCALE=7.5
export POSE_STEPS=30
export POSE_INFUSENET_COND_SCALE=1.0
export POSE_INFUSENET_GUIDANCE_START=0.0
export POSE_INFUSENET_GUIDANCE_END=1.0
export POSE_ENABLE_REALISM="true"
export POSE_ENABLE_ANTI_BLUR="false"
export POSE_MODEL_VERSION="sim_stage1"
export QUANTIZE_8BIT="true"
export CPU_OFFLOAD="true"
export FLUX_MODEL_PATH="/workspace/stonevol/ext_models/FLUX.1-dev"
export INFINITE_MODEL_PATH="/workspace/stonevol/ext_models/InfiniteYou"

export OUTPUT_PATH="/workspace/stonevol/infU_samples"
export LIGHT_IMG_PATH="/workspace/stonevol/infU_samples/pose.png"
export LIGHT_PROMPT="A beautiful sunset over the ocean"
export LIGHT_NUM_SAMPLES=1
export LIGHT_SEED=67890
export LIGHT_STEPS=25
export LIGHT_A_PROMPT="best quality"
export LIGHT_N_PROMPT="lowres, bad anatomy, bad hands, cropped, worst quality"
export LIGHT_CFG=2.0
export LIGHT_HIGHRES_SCALE=2.0
export LIGHT_HIGHRES_DENOISE=0.75
export LIGHT_BG_SOURCE="None"
export LIGHT_LOWRES_DENOISE=0.9

python synthesize_pose.py --input_img $INPUT_IMAGE --pose_prompt "$POSE_PROMPT" --pose_seed $POSE_SEED \
--pose_guidance_scale $POSE_GUIDANCE_SCALE --pose_steps $POSE_STEPS --pose_infusenet_cond_scale $POSE_INFUSENET_COND_SCALE \
--pose_infusenet_guidance_start $POSE_INFUSENET_GUIDANCE_START --pose_infusenet_guidance_end $POSE_INFUSENET_GUIDANCE_END \
--pose_enable_realism $POSE_ENABLE_REALISM --pose_enable_anti_blur $POSE_ENABLE_ANTI_BLUR --pose_model_version $POSE_MODEL_VERSION \
--quantize_8bit $QUANTIZE_8BIT --cpu_offload $CPU_OFFLOAD --flux_model_path $FLUX_MODEL_PATH --infinite_model_path $INFINITE_MODEL_PATH \
--out_path $OUTPUT_PATH


# python ic_light.py --input_img $LIGHT_IMG_PATH --out_path $OUTPUT_PATH --light_prompt "$LIGHT_PROMPT" --light_num_samples $LIGHT_NUM_SAMPLES --light_seed $LIGHT_SEED \
# --light_steps $LIGHT_STEPS --light_a_prompt "$LIGHT_A_PROMPT" --light_n_prompt "$LIGHT_N_PROMPT" \
# --light_cfg $LIGHT_CFG --light_highres_scale $LIGHT_HIGHRES_SCALE --light_highres_denoise $LIGHT_HIGHRES_DENOISE \
# --light_lowres_denoise $LIGHT_LOWRES_DENOISE --light_bg_source $LIGHT_BG_SOURCE