# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

model="/workplace/models/Dream/Dream-v0-Instruct-7B"
model_name="Dream-v0-Instruct-7B"
device=1

############################################### gsm8k evaluations ###############################################
task=gsm8k
length=256
block_length=32
num_fewshot=5
steps=$((length / block_length))

# # dsb + cache + parallel
# CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,dsb=true,use_cache=true,prefix_cache=0,show_speed=True,outp_path=evals_results_${model_name}_pfx/dsb_cache_parallel/${task}-ns0-${length}/results.jsonl \
#     --tasks ${task} \
#     --num_fewshot ${num_fewshot} \
#     --batch_size 1 \
#     --output_path evals_results_${model_name}_pfx/dsb_cache_parallel/${task}-ns0-${length}

# # # dsb + cache + parallel
# CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,dsb=true,use_cache=true,prefix_cache=4,show_speed=True,outp_path=evals_results_${model_name}_pfx/dsb_cache_parallel/${task}-ns0-${length}/results.jsonl \
#     --tasks ${task} \
#     --num_fewshot ${num_fewshot} \
#     --batch_size 1 \
#     --output_path evals_results_${model_name}_pfx/dsb_cache_parallel/${task}-ns0-${length}

# # # dsb + cache + parallel
# CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,dsb=true,use_cache=true,prefix_cache=8,show_speed=True,outp_path=evals_results_${model_name}_pfx/dsb_cache_parallel/${task}-ns0-${length}/results.jsonl \
#     --tasks ${task} \
#     --num_fewshot ${num_fewshot} \
#     --batch_size 1 \
#     --output_path evals_results_${model_name}_pfx/dsb_cache_parallel/${task}-ns0-${length}

# # dsb + cache + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,dsb=true,use_cache=true,prefix_cache=12,show_speed=True,outp_path=evals_results_${model_name}_pfx/dsb_cache_parallel/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --output_path evals_results_${model_name}_pfx/dsb_cache_parallel/${task}-ns0-${length}

# # dsb + cache + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,dsb=true,use_cache=true,prefix_cache=16,show_speed=True,outp_path=evals_results_${model_name}_pfx/dsb_cache_parallel/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --output_path evals_results_${model_name}_pfx/dsb_cache_parallel/${task}-ns0-${length}

# # dsb + cache + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,dsb=true,use_cache=true,prefix_cache=20,show_speed=True,outp_path=evals_results_${model_name}_pfx/dsb_cache_parallel/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --output_path evals_results_${model_name}_pfx/dsb_cache_parallel/${task}-ns0-${length}

# # dsb + cache + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,dsb=true,use_cache=true,prefix_cache=24,show_speed=True,outp_path=evals_results_${model_name}_pfx/dsb_cache_parallel/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --output_path evals_results_${model_name}_pfx/dsb_cache_parallel/${task}-ns0-${length}

# # dsb + cache + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,dsb=true,use_cache=true,prefix_cache=28,show_speed=True,outp_path=evals_results_${model_name}_pfx/dsb_cache_parallel/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --output_path evals_results_${model_name}_pfx/dsb_cache_parallel/${task}-ns0-${length}

# # dsb + cache + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,dsb=true,use_cache=true,prefix_cache=32,show_speed=True,outp_path=evals_results_${model_name}_pfx/dsb_cache_parallel/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --output_path evals_results_${model_name}_pfx/dsb_cache_parallel/${task}-ns0-${length}

