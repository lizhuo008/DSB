# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true


############################################### gsm8k evaluations ###############################################
task=gsm8k
length=256
block_length=32
num_fewshot=5
steps=$((length / block_length))
model_path='/workplace/models/LLaDA/LLaDA-8B-Instruct'

# baseline
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,outp_path=evals_results/baseline/${task}-ns0-${length}/results.jsonl \
# --output_path evals_results/baseline/${task}-ns0-${length}

# parallel
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.9,outp_path=evals_results/parallel/${task}-ns0-${length}/results.jsonl \
# --output_path evals_results/parallel/${task}-ns0-${length}

# ib + parallel
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,ib=True,threshold=0.9,show_speed=True,outp_path=evals_results/ib_parallel/${task}-ns0-${length}/results.jsonl \
# --output_path evals_results/ib_parallel/${task}-ns0-${length}

# dual cache + parallel
CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,dual_cache=True,threshold=0.9,show_speed=True,outp_path=evals_results/dual_cache_parallel/${task}-ns0-${length}/results.jsonl \
--output_path evals_results/dual_cache_parallel/${task}-ns0-${length}

# ib_cache + parallel
CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,ib=True,threshold=0.9,show_speed=True,outp_path=evals_results/ib_cache_parallel/${task}-ns0-${length}/results.jsonl \
--output_path evals_results/ib_cache_parallel/${task}-ns0-${length}
############################################### minerva_math evaluations ###############################################
task=minerva_math
length=256
block_length=32
num_fewshot=4
steps=$((length / block_length))
model_path='/workplace/models/LLaDA/LLaDA-8B-Instruct'

# baseline
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,outp_path=evals_results/baseline/${task}-ns0-${length}/results.jsonl \
# --output_path evals_results/baseline/${task}-ns0-${length}

# parallel
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.9,outp_path=evals_results/parallel/${task}-ns0-${length}/results.jsonl \
# --output_path evals_results/parallel/${task}-ns0-${length}

# ib + parallel
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,ib=True,threshold=0.9,show_speed=True,outp_path=evals_results/ib_parallel/${task}-ns0-${length}/results.jsonl \
# --output_path evals_results/ib_parallel/${task}-ns0-${length}

# dual cache + parallel
CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,dual_cache=True,threshold=0.9,show_speed=True,outp_path=evals_results/dual_cache_parallel/${task}-ns0-${length}/results.jsonl \
--output_path evals_results/dual_cache_parallel/${task}-ns0-${length}

# ib_cache + parallel
CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,ib=True,threshold=0.9,show_speed=True,outp_path=evals_results/ib_cache_parallel/${task}-ns0-${length}/results.jsonl \
--output_path evals_results/ib_cache_parallel/${task}-ns0-${length}


# ############################################### humaneval evaluations ###############################################
task=humaneval
length=256
block_length=32
num_fewshot=0
steps=$((length / block_length))
model_path='/workplace/models/LLaDA-8B-Instruct'

# baseline
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,outp_path=evals_results/baseline/${task}-ns0-${length}/results.jsonl \
# --output_path evals_results/baseline/${task}-ns0-${length} --log_samples

# parallel
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.9,outp_path=evals_results/parallel/${task}-ns0-${length}/results.jsonl \
# --output_path evals_results/parallel/${task}-ns0-${length} --log_samples

# ib + parallel
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,ib=True,threshold=0.9,show_speed=True,outp_path=evals_results/ib_parallel/${task}-ns0-${length}/results.jsonl \
# --output_path evals_results/ib_parallel/${task}-ns0-${length} --log_samples

# dual cache + parallel
CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,dual_cache=True,threshold=0.9,show_speed=True,outp_path=evals_results/dual_cache_parallel/${task}-ns0-${length}/results.jsonl \
--output_path evals_results/dual_cache_parallel/${task}-ns0-${length} --log_samples

# ib_cache + parallel
CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,ib=True,threshold=0.9,show_speed=True,outp_path=evals_results/ib_cache_parallel/${task}-ns0-${length}/results.jsonl \
--output_path evals_results/ib_cache_parallel/${task}-ns0-${length} --log_samples

# ############################################### mbpp evaluations ###############################################
task=mbpp
length=256
block_length=32
num_fewshot=3
steps=256
model_path='/workplace/models/LLaDA-8B-Instruct'

# baseline
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,outp_path=evals_results/baseline/${task}-ns0-${length}/results.jsonl \
# --output_path evals_results/baseline/${task}-ns0-${length} --log_samples

# parallel
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.9,outp_path=evals_results/parallel/${task}-ns0-${length}/results.jsonl \
# --output_path evals_results/parallel/${task}-ns0-${length} --log_samples

# ib + parallel
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,ib=True,threshold=0.9,show_speed=True,outp_path=evals_results/ib_parallel/${task}-ns0-${length}/results.jsonl \
# --output_path evals_results/ib_parallel/${task}-ns0-${length} --log_samples

# dual cache + parallel
CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,dual_cache=True,threshold=0.9,show_speed=True,outp_path=evals_results/dual_cache_parallel/${task}-ns0-${length}/results.jsonl \
--output_path evals_results/dual_cache_parallel/${task}-ns0-${length} --log_samples

# ib_cache + parallel
CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,ib=True,threshold=0.9,show_speed=True,outp_path=evals_results/ib_cache_parallel/${task}-ns0-${length}/results.jsonl \
--output_path evals_results/ib_cache_parallel/${task}-ns0-${length} --log_samples
