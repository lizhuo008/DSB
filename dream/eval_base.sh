# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

model="Dream-org/Dream-v0-Base-7B"
model_name="Dream-v0-Base-7B"
device=0

length=256
block_length=32

############################################### gsm8k evaluations ###############################################
task=gsm8k
num_fewshot=5
steps=$((length / block_length))


# baseline
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},block_length=${block_length},add_bos_token=true,alg=entropy,show_speed=True,outp_path=evals_results_${model_name}/baseline/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/baseline/${task}-ns0-${length} --log_samples


# naive + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,show_speed=True,outp_path=evals_results_${model_name}/naive_parallel/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/naive_parallel/${task}-ns0-${length} --log_samples

# # dsb + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,dsb=true,show_speed=True,outp_path=evals_results_${model_name}/dsb_parallel/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/dsb_parallel/${task}-ns0-${length} --log_samples

# dsb + parallel + constant
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,dsb=true,max_block_length=${block_length},show_speed=True,outp_path=evals_results_${model_name}/dsb_parallel_constant/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/dsb_parallel_constant/${task}-ns0-${length} --log_samples

# dual cache + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,use_cache=true,dual_cache=true,show_speed=True,outp_path=evals_results_${model_name}/dual_cache_parallel/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/dual_cache_parallel/${task}-ns0-${length} --log_samples

# # dsb + cache + parallel
CUDA_VISIBLE_DEVICES=0 accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,prefix_window=4,dsb=true,use_cache=true,show_speed=True,outp_path=evals_results_${model_name}/dsb_cache_parallel/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/dsb_cache_parallel/${task}-ns0-${length} --log_samples

# dsb + cache + parallel + constant
CUDA_VISIBLE_DEVICES=0 accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,prefix_window=4,dsb=true,use_cache=true,max_block_length=${block_length},show_speed=True,outp_path=evals_results_${model_name}/dsb_cache_parallel_constant/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/dsb_cache_parallel_constant/${task}-ns0-${length} --log_samples

############################################### minerva_math evaluations ###############################################
task=minerva_math
num_fewshot=4
steps=$((length / block_length))

# baseline
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},block_length=${block_length},add_bos_token=true,alg=entropy,show_speed=True,outp_path=evals_results_${model_name}/baseline/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/baseline/${task}-ns0-${length} --log_samples


# naive + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,show_speed=True,outp_path=evals_results_${model_name}/naive_parallel/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/naive_parallel/${task}-ns0-${length} --log_samples

# # dsb + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,dsb=true,show_speed=True,outp_path=evals_results_${model_name}/dsb_parallel/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/dsb_parallel/${task}-ns0-${length} --log_samples

# dsb + parallel + constant
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,dsb=true,max_block_length=${block_length},show_speed=True,outp_path=evals_results_${model_name}/dsb_parallel_constant/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/dsb_parallel_constant/${task}-ns0-${length} --log_samples

# dual cache + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,use_cache=true,dual_cache=true,show_speed=True,outp_path=evals_results_${model_name}/dual_cache_parallel/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/dual_cache_parallel/${task}-ns0-${length} --log_samples

# # dsb + cache + parallel
CUDA_VISIBLE_DEVICES=0 accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,prefix_window=4,dsb=true,use_cache=true,show_speed=True,outp_path=evals_results_${model_name}/dsb_cache_parallel/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/dsb_cache_parallel/${task}-ns0-${length} --log_samples

# dsb + cache + parallel + constant
CUDA_VISIBLE_DEVICES=0 accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,prefix_window=4,dsb=true,use_cache=true,max_block_length=${block_length},show_speed=True,outp_path=evals_results_${model_name}/dsb_cache_parallel_constant/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/dsb_cache_parallel_constant/${task}-ns0-${length} --log_samples

############################################### humaneval evaluations ###############################################
task=humaneval
num_fewshot=0
steps=$((length / block_length))

# baseline
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},block_length=${block_length},add_bos_token=true,alg=entropy,show_speed=True,outp_path=evals_results_${model_name}/baseline/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/baseline/${task}-ns0-${length} --log_samples


# naive + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,show_speed=True,outp_path=evals_results_${model_name}/naive_parallel/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/naive_parallel/${task}-ns0-${length} --log_samples

# # dsb + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,dsb=true,show_speed=True,outp_path=evals_results_${model_name}/dsb_parallel/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/dsb_parallel/${task}-ns0-${length} --log_samples

# dsb + parallel + constant
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,dsb=true,max_block_length=${block_length},show_speed=True,outp_path=evals_results_${model_name}/dsb_parallel_constant/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/dsb_parallel_constant/${task}-ns0-${length} --log_samples

# dual cache + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,use_cache=true,dual_cache=true,show_speed=True,outp_path=evals_results_${model_name}/dual_cache_parallel/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/dual_cache_parallel/${task}-ns0-${length} --log_samples

# # dsb + cache + parallel
CUDA_VISIBLE_DEVICES=0 accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,prefix_window=4,dsb=true,use_cache=true,show_speed=True,outp_path=evals_results_${model_name}/dsb_cache_parallel/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/dsb_cache_parallel/${task}-ns0-${length} --log_samples

# dsb + cache + parallel + constant
CUDA_VISIBLE_DEVICES=0 accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,prefix_window=4,dsb=true,use_cache=true,max_block_length=${block_length},show_speed=True,outp_path=evals_results_${model_name}/dsb_cache_parallel_constant/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/dsb_cache_parallel_constant/${task}-ns0-${length} --log_samples

############################################### mbpp evaluations ###############################################
task=mbpp
num_fewshot=3
steps=$((length / block_length))

# baseline
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},block_length=${block_length},add_bos_token=true,alg=entropy,show_speed=True,outp_path=evals_results_${model_name}/baseline/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/baseline/${task}-ns0-${length} --log_samples


# naive + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,show_speed=True,outp_path=evals_results_${model_name}/naive_parallel/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/naive_parallel/${task}-ns0-${length} --log_samples

# # dsb + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,dsb=true,show_speed=True,outp_path=evals_results_${model_name}/dsb_parallel/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/dsb_parallel/${task}-ns0-${length} --log_samples

# dsb + parallel + constant
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,dsb=true,max_block_length=${block_length},show_speed=True,outp_path=evals_results_${model_name}/dsb_parallel_constant/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/dsb_parallel_constant/${task}-ns0-${length} --log_samples

# dual cache + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,use_cache=true,dual_cache=true,show_speed=True,outp_path=evals_results_${model_name}/dual_cache_parallel/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/dual_cache_parallel/${task}-ns0-${length} --log_samples

# # dsb + cache + parallel
CUDA_VISIBLE_DEVICES=0 accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,prefix_window=4,dsb=true,use_cache=true,show_speed=True,outp_path=evals_results_${model_name}/dsb_cache_parallel/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/dsb_cache_parallel/${task}-ns0-${length} --log_samples

# dsb + cache + parallel + constant
CUDA_VISIBLE_DEVICES=0 accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,prefix_window=4,dsb=true,use_cache=true,max_block_length=${block_length},show_speed=True,outp_path=evals_results_${model_name}/dsb_cache_parallel_constant/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/dsb_cache_parallel_constant/${task}-ns0-${length} --log_samples

############################################### bhh evaluations ###############################################
task=bbh
num_fewshot=3
steps=$((length / block_length))

# baseline
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},block_length=${block_length},add_bos_token=true,alg=entropy,show_speed=True,outp_path=evals_results_${model_name}/baseline/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/baseline/${task}-ns0-${length} --log_samples


# naive + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,show_speed=True,outp_path=evals_results_${model_name}/naive_parallel/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/naive_parallel/${task}-ns0-${length} --log_samples

# # dsb + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,dsb=true,show_speed=True,outp_path=evals_results_${model_name}/dsb_parallel/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/dsb_parallel/${task}-ns0-${length} --log_samples

# dsb + parallel + constant
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,dsb=true,max_block_length=${block_length},show_speed=True,outp_path=evals_results_${model_name}/dsb_parallel_constant/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/dsb_parallel_constant/${task}-ns0-${length} --log_samples

# dual cache + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,use_cache=true,dual_cache=true,show_speed=True,outp_path=evals_results_${model_name}/dual_cache_parallel/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/dual_cache_parallel/${task}-ns0-${length} --log_samples

# # dsb + cache + parallel
CUDA_VISIBLE_DEVICES=0 accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,prefix_window=4,dsb=true,use_cache=true,show_speed=True,outp_path=evals_results_${model_name}/dsb_cache_parallel/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/dsb_cache_parallel/${task}-ns0-${length} --log_samples

# dsb + cache + parallel + constant
CUDA_VISIBLE_DEVICES=0 accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,prefix_window=4,dsb=true,use_cache=true,max_block_length=${block_length},show_speed=True,outp_path=evals_results_${model_name}/dsb_cache_parallel_constant/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/dsb_cache_parallel_constant/${task}-ns0-${length} --log_samples
