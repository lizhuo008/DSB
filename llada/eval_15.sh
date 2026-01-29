# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

device=0

length=256
block_length=32
model_path='GSAI-ML/LLaDA-1.5'
model_name='LLaDA-1.5'

############################################### gsm8k evaluations ###############################################
task=gsm8k
num_fewshot=5
steps=$((length / block_length))

# baseline
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,outp_path=evals_results_${model_name}/baseline/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/baseline/${task}-ns0-${length}

# parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.9,outp_path=evals_results_${model_name}/parallel/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/parallel/${task}-ns0-${length}

# dsb + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,dsb=True,threshold=0.9,max_block_length=-1,show_speed=True,outp_path=evals_results_${model_name}/dsb_parallel/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/dsb_parallel/${task}-ns0-${length}

# dsb + parallel + constant
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,dsb=True,threshold=0.9,max_block_length=${block_length},show_speed=True,outp_path=evals_results_${model_name}/dsb_parallel_constant/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/dsb_parallel_constant/${task}-ns0-${length}

# dual cache + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,dual_cache=True,threshold=0.9,show_speed=True,outp_path=evals_results_${model_name}/dual_cache_parallel/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/dual_cache_parallel/${task}-ns0-${length}

# dsb_cache + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,dsb=True,threshold=0.9,show_speed=True,pwl=24,swl=0,outp_path=evals_results_${model_name}/dsb_cache_parallel/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/dsb_cache_parallel/${task}-ns0-${length}

# dsb_cache + parallel + constant
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,dsb=True,threshold=0.9,show_speed=True,pwl=24,swl=0,max_block_length=${block_length},show_speed=True,outp_path=evals_results_${model_name}/dsb_cache_parallel_constant/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/dsb_cache_parallel_constant/${task}-ns0-${length}
############################################### minerva_math evaluations ###############################################
task=minerva_math
num_fewshot=4
steps=$((length / block_length))

# baseline
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,outp_path=evals_results_${model_name}/baseline/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/baseline/${task}-ns0-${length}

# parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.9,outp_path=evals_results_${model_name}/parallel/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/parallel/${task}-ns0-${length}

# dsb + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,dsb=True,threshold=0.9,max_block_length=-1,show_speed=True,outp_path=evals_results_${model_name}/dsb_parallel/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/dsb_parallel/${task}-ns0-${length}

# dsb + parallel + constant
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,dsb=True,threshold=0.9,max_block_length=${block_length},show_speed=True,outp_path=evals_results_${model_name}/dsb_parallel_constant/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/dsb_parallel_constant/${task}-ns0-${length}

# dual cache + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,dual_cache=True,threshold=0.9,show_speed=True,outp_path=evals_results_${model_name}/dual_cache_parallel/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/dual_cache_parallel/${task}-ns0-${length}

# dsb_cache + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,dsb=True,threshold=0.9,show_speed=True,pwl=24,swl=0,outp_path=evals_results_${model_name}/dsb_cache_parallel/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/dsb_cache_parallel/${task}-ns0-${length}

# dsb_cache + parallel + constant
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,dsb=True,threshold=0.9,show_speed=True,pwl=24,swl=0,max_block_length=${block_length},show_speed=True,outp_path=evals_results_${model_name}/dsb_cache_parallel_constant/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/dsb_cache_parallel_constant/${task}-ns0-${length}


# ############################################### humaneval evaluations ###############################################
task=humaneval
num_fewshot=0
steps=$((length / block_length))

# baseline
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,outp_path=evals_results_${model_name}/baseline/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/baseline/${task}-ns0-${length} --log_samples

# parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.9,outp_path=evals_results_${model_name}/parallel/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/parallel/${task}-ns0-${length} --log_samples

# dsb + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,dsb=True,threshold=0.9,max_block_length=-1,show_speed=True,outp_path=evals_results_${model_name}/dsb_parallel/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/dsb_parallel/${task}-ns0-${length} --log_samples

# dsb + parallel + constant
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,dsb=True,threshold=0.9,max_block_length=${block_length},show_speed=True,outp_path=evals_results_${model_name}/dsb_parallel_constant/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/dsb_parallel_constant/${task}-ns0-${length} --log_samples

# dual cache + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,dual_cache=True,threshold=0.9,show_speed=True,outp_path=evals_results_${model_name}/dual_cache_parallel/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/dual_cache_parallel/${task}-ns0-${length} --log_samples

# dsb_cache + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,dsb=True,threshold=0.9,show_speed=True,pwl=24,swl=0,outp_path=evals_results_${model_name}/dsb_cache_parallel/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/dsb_cache_parallel/${task}-ns0-${length} --log_samples

# dsb_cache + parallel + constant
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,dsb=True,threshold=0.9,show_speed=True,pwl=24,swl=0,max_block_length=${block_length},show_speed=True,outp_path=evals_results_${model_name}/dsb_cache_parallel_constant/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/dsb_cache_parallel_constant/${task}-ns0-${length} --log_samples

# ############################################### mbpp evaluations ###############################################
task=mbpp
num_fewshot=3
steps=$((length / block_length))

# baseline
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,outp_path=evals_results_${model_name}/baseline/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/baseline/${task}-ns0-${length} --log_samples

# parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.9,outp_path=evals_results_${model_name}/parallel/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/parallel/${task}-ns0-${length} --log_samples

# dsb + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,dsb=True,threshold=0.9,max_block_length=-1,show_speed=True,outp_path=evals_results_${model_name}/dsb_parallel/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/dsb_parallel/${task}-ns0-${length} --log_samples

# dsb + parallel + constant
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,dsb=True,threshold=0.9,max_block_length=${block_length},show_speed=True,outp_path=evals_results_${model_name}/dsb_parallel_constant/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/dsb_parallel_constant/${task}-ns0-${length} --log_samples

# dual cache + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,dual_cache=True,threshold=0.9,show_speed=True,outp_path=evals_results_${model_name}/dual_cache_parallel/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/dual_cache_parallel/${task}-ns0-${length} --log_samples

# dsb_cache + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,dsb=True,threshold=0.9,show_speed=True,pwl=24,swl=0,outp_path=evals_results_${model_name}/dsb_cache_parallel/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/dsb_cache_parallel/${task}-ns0-${length} --log_samples

# dsb_cache + parallel + constant
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,dsb=True,threshold=0.9,show_speed=True,pwl=24,swl=0,max_block_length=${block_length},show_speed=True,outp_path=evals_results_${model_name}/dsb_cache_parallel_constant/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/dsb_cache_parallel_constant/${task}-ns0-${length} --log_samples


# ############################################### bbh evaluations ###############################################
task=bbh
num_fewshot=3
steps=$((length / block_length))

# baseline
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,outp_path=evals_results_${model_name}/baseline/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/baseline/${task}-ns0-${length}

# parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.9,outp_path=evals_results_${model_name}/parallel/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/parallel/${task}-ns0-${length}

# dsb + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,dsb=True,threshold=0.9,max_block_length=-1,show_speed=True,outp_path=evals_results_${model_name}/dsb_parallel/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/dsb_parallel/${task}-ns0-${length}

# dsb + parallel + constant
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,dsb=True,threshold=0.9,max_block_length=${block_length},show_speed=True,outp_path=evals_results_${model_name}/dsb_parallel_constant/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/dsb_parallel_constant/${task}-ns0-${length}

# dual cache + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,dual_cache=True,threshold=0.9,show_speed=True,outp_path=evals_results_${model_name}/dual_cache_parallel/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/dual_cache_parallel/${task}-ns0-${length}

# dsb_cache + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,dsb=True,threshold=0.9,show_speed=True,pwl=24,swl=0,outp_path=evals_results_${model_name}/dsb_cache_parallel/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/dsb_cache_parallel/${task}-ns0-${length}

# dsb_cache + parallel + constant
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,dsb=True,threshold=0.9,show_speed=True,pwl=24,swl=0,max_block_length=${block_length},show_speed=True,outp_path=evals_results_${model_name}/dsb_cache_parallel_constant/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/dsb_cache_parallel_constant/${task}-ns0-${length}