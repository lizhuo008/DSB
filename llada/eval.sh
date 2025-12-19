# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true


############################################### gsm8k evaluations ###############################################
task=gsm8k
length=256
block_length=32
num_fewshot=5
limit=3
steps=$((length / block_length))
model_path='/workplace/models/LLaDA-8B-Instruct'
factor=1

# baseline
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True \
# --output_path evals_results/baseline/gsm8k-ns0-${length}

# parallel
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} --limit ${limit} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.7 \
# --output_path evals_results/parallel/gsm8k-ns0-${length}

# # factor
# CUDA_VISIBLE_DEVICES=1 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} --limit ${limit} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,factor=${factor} \
# --output_path evals_results/factor/gsm8k-ns0-${length}

# dual cache + parallel factor
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} --limit ${limit} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,dual_cache=True,factor=${factor},show_speed=True \
# --output_path evals_results/dual_cache_parallel/gsm8k-ns0-${length}


# sb
# CUDA_VISIBLE_DEVICES=1 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,sb=True,factor=${factor} \
# --output_path evals_results/sb/gsm8k-ns0-${length}

# ib
# CUDA_VISIBLE_DEVICES=1 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} --limit ${limit} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,ib=True,factor=${factor} \
# --output_path evals_results/ib/gsm8k-ns0-${length}

# ib_cache
CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,use_cache=True,ib=True,factor=${factor} \
--output_path evals_results/ib_cache/gsm8k-ns0-${length}

############################################### minerva_math evaluations ###############################################
# task=minerva_math
# length=256
# block_length=32
# num_fewshot=4
# steps=256

# # baseline
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29600 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path='/workplace/models/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,task="minerva_math"

# # dParallel
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29601 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path='/workplace/models/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,task="minerva_math"



# ############################################### humaneval evaluations ###############################################
# task=humaneval
# length=256
# block_length=32
# num_fewshot=0
# steps=256

# # baseline
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29600 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path='/workplace/models/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,task="humaneval" \
# --output_path evals_results/baseline/humaneval-ns0-${length} --log_samples

# # dparallel
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29601 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path='/workplace/models/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,task="humaneval" \
# --output_path evals_results/parallel/humaneval-ns0-${length} --log_samples

# ## NOTICE: use postprocess for humaneval
# python postprocess_code_humaneval.py {the samples_xxx.jsonl file under output_path}





# ############################################### mbpp evaluations ###############################################
# task=mbpp
# length=256
# block_length=32
# num_fewshot=3
# steps=256

# # baseline
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29600 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path='/workplace/models/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,task="mbpp" \
# --output_path evals_results/baseline/mbpp-ns0-${length} --log_samples

# # parallel
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29601 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path='/workplace/models/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,task="mbpp" \
# --output_path evals_results/parallel/mbpp-ns0-${length} --log_samples

# ## NOTICE: use postprocess for mbpp
# python postprocess_code_mbpp.py {the samples_xxx.jsonl file under output_path}
