# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

model="Dream-org/Dream-v0-Instruct-7B"
model_name="Dream-v0-Instruct-7B"

device=0

length=256
block_length=32

num_fewshot=3
steps=$((length / block_length))

for task in \
  minerva_math_algebra \
  minerva_math_counting_and_probability \
  minerva_math_geometry \
  minerva_math_intermediate_algebra \
  minerva_math_num_theory \
  minerva_math_prealgebra \
  minerva_math_precalc 
do
    # baseline
    CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},block_length=${block_length},add_bos_token=true,alg=entropy,show_speed=True,outp_path=evals_results_${model_name}/baseline/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/baseline/${task}-ns0-${length} --log_samples
done