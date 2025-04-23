#/bin/bash
export INPUT_PATH="/data/org/100831"
export OUTPUT_PATH="/data/cropped/100831"
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

python ic_light.py --input_dir $INPUT_PATH --out_path $OUTPUT_PATH --light_prompt "$LIGHT_PROMPT" --light_num_samples $LIGHT_NUM_SAMPLES --light_seed $LIGHT_SEED \
--light_steps $LIGHT_STEPS --light_a_prompt "$LIGHT_A_PROMPT" --light_n_prompt "$LIGHT_N_PROMPT" \
--light_cfg $LIGHT_CFG --light_highres_scale $LIGHT_HIGHRES_SCALE --light_highres_denoise $LIGHT_HIGHRES_DENOISE \
--light_lowres_denoise $LIGHT_LOWRES_DENOISE --light_bg_source $LIGHT_BG_SOURCE