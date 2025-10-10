#/bin/bash
export INPUT_PATH="/workspace/datasetvol/mvhuman_data/mv_captures"
export OUTPUT_PATH="/workspace/datasetvol/mvhuman_data/relit_images"
export LIGHT_NUM_SAMPLES=1
export LIGHT_SEED=67890
export LIGHT_STEPS=25
export LIGHT_A_PROMPT="best quality"
export LIGHT_N_PROMPT="lowres, bad anatomy, bad hands, cropped, worst quality, pitch black, multiple people"
export LIGHT_CFG=2.0
export LIGHT_HIGHRES_SCALE=1.77777777778 # 1024/576
export LIGHT_HIGHRES_DENOISE=0.75
export LIGHT_BG_SOURCE="None"
export LIGHT_LOWRES_DENOISE=0.9
export STEP_SIZE=60
# export BLACKLIST_FILE="/workspace/stonevol/subjects_to_rerun_iclight.txt"

FORCE_FLAG=""
for arg in "$@"; do
case "$arg" in 
--force|-f) FORCE_FLAG="--force" ;;
esac
done

python ic_light.py --input_dir $INPUT_PATH --out_path $OUTPUT_PATH --light_num_samples $LIGHT_NUM_SAMPLES --light_seed $LIGHT_SEED \
--light_steps $LIGHT_STEPS --light_a_prompt "$LIGHT_A_PROMPT" --light_n_prompt "$LIGHT_N_PROMPT" \
--light_cfg $LIGHT_CFG --light_highres_scale $LIGHT_HIGHRES_SCALE --light_highres_denoise $LIGHT_HIGHRES_DENOISE \
--light_lowres_denoise $LIGHT_LOWRES_DENOISE --light_bg_source $LIGHT_BG_SOURCE --step_size $STEP_SIZE --padding 60 $FORCE_FLAG
