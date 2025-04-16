#/bin/bash

export INPUT_IMAGE="/mnt/d/consistent-3d-generation/samples/input1/CC32871A017_cropped_0220.png"
export POSE_PROMPT="a woman stretching her arms"
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

export OUTPUT_PATH="/mnt/d/consistent-3d-generation/samples/synthesized_images"
export LIGHT_PROMPT="A beautiful sunset over the ocean"
export LIGHT_NUM_SAMPLES=5
export LIGHT_SEED=67890
export LIGHT_STEPS=50
export LIGHT_A_PROMPT="best quality"
export LIGHT_N_PROMPT="lowres, bad anatomy, bad hands, cropped, worst quality"
export LIGHT_CFG=7.5
export LIGHT_HIGHRES_SCALE=2.0
export LIGHT_HIGHRES_DENOISE=0.75
export LIGHT_BG_SOURCE="None"

python synthesize.py --input_img $INPUT_IMAGE --pose_prompt "$POSE_PROMPT" --pose_seed $POSE_SEED \
--pose_guidance_scale $POSE_GUIDANCE_SCALE --pose_steps $POSE_STEPS --pose_infusenet_cond_scale $POSE_INFUSENET_COND_SCALE \
--pose_infusenet_guidance_start $POSE_INFUSENET_GUIDANCE_START --pose_infusenet_guidance_end $POSE_INFUSENET_GUIDANCE_END \
--pose_enable_realism $POSE_ENABLE_REALISM --pose_enable_anti_blur $POSE_ENABLE_ANTI_BLUR --pose_model_version $POSE_MODEL_VERSION \
--quantize_8bit $QUANTIZE_8BIT --cpu_offload $CPU_OFFLOAD \
--out_path $OUTPUT_PATH --light_prompt "$LIGHT_PROMPT" --light_num_samples $LIGHT_NUM_SAMPLES --light_seed $LIGHT_SEED \
--light_steps $LIGHT_STEPS --light_a_prompt "$LIGHT_A_PROMPT" --light_n_prompt "$LIGHT_N_PROMPT" \
--light_cfg $LIGHT_CFG --light_highres_scale $LIGHT_HIGHRES_SCALE --light_highres_denoise $LIGHT_HIGHRES_DENOISE \
--light_bg_source $LIGHT_BG_SOURCE