# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

device=1

length=256
block_length=32
model_path='/workplace/models/LLaDA/LLaDA-8B-Instruct'
model_name='LLaDA-8B-Instruct'
# model_path='/workplace/models/LLaDA/LLaDA-1.5'
# model_name='LLaDA-1.5'

############################################### gsm8k evaluations ###############################################
task=gsm8k
num_fewshot=5
steps=$((length / block_length))

# ib_cache + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,ib=True,threshold=0.9,show_speed=True,pwl=0,swl=0,outp_path=evals_results_${model_name}_sfx/ib_cache_parallel/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}_sfx/ib_cache_parallel/${task}-ns0-${length}

# ib_cache + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,ib=True,threshold=0.9,show_speed=True,pwl=0,swl=4,outp_path=evals_results_${model_name}_sfx/ib_cache_parallel/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}_sfx/ib_cache_parallel/${task}-ns0-${length}

# ib_cache + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,ib=True,threshold=0.9,show_speed=True,pwl=0,swl=8,outp_path=evals_results_${model_name}_sfx/ib_cache_parallel/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}_sfx/ib_cache_parallel/${task}-ns0-${length}

# ib_cache + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,ib=True,threshold=0.9,show_speed=True,pwl=0,swl=12,outp_path=evals_results_${model_name}_sfx/ib_cache_parallel/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}_sfx/ib_cache_parallel/${task}-ns0-${length}

# ib_cache + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,ib=True,threshold=0.9,show_speed=True,pwl=0,swl=16,outp_path=evals_results_${model_name}_sfx/ib_cache_parallel/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}_sfx/ib_cache_parallel/${task}-ns0-${length}

# ib_cache + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,ib=True,threshold=0.9,show_speed=True,pwl=0,swl=20,outp_path=evals_results_${model_name}_sfx/ib_cache_parallel/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}_sfx/ib_cache_parallel/${task}-ns0-${length}

# ib_cache + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,ib=True,threshold=0.9,show_speed=True,pwl=0,swl=24,outp_path=evals_results_${model_name}_sfx/ib_cache_parallel/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}_sfx/ib_cache_parallel/${task}-ns0-${length}

# ib_cache + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,ib=True,threshold=0.9,show_speed=True,pwl=0,swl=28,outp_path=evals_results_${model_name}_sfx/ib_cache_parallel/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}_sfx/ib_cache_parallel/${task}-ns0-${length}

# ib_cache + parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,ib=True,threshold=0.9,show_speed=True,pwl=0,swl=32,outp_path=evals_results_${model_name}_sfx/ib_cache_parallel/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}_sfx/ib_cache_parallel/${task}-ns0-${length}