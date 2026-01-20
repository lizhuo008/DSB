# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

model="Dream-org/Dream-v0-Base-7B"
model_name="Dream-v0-Base-7B"
device=0

length=256
block_length=32

num_fewshot=3
steps=$((length / block_length))

for task in \
  bbh_cot_fewshot \
  bbh_cot_fewshot_boolean_expressions \
  bbh_cot_fewshot_causal_judgement \
  bbh_cot_fewshot_date_understanding \
  bbh_cot_fewshot_disambiguation_qa \
  bbh_cot_fewshot_dyck_languages \
  bbh_cot_fewshot_formal_fallacies \
  bbh_cot_fewshot_geometric_shapes \
  bbh_cot_fewshot_hyperbaton \
  bbh_cot_fewshot_logical_deduction_five_objects \
  bbh_cot_fewshot_logical_deduction_seven_objects \
  bbh_cot_fewshot_logical_deduction_three_objects \
  bbh_cot_fewshot_movie_recommendation \
  bbh_cot_fewshot_multistep_arithmetic_two \
  bbh_cot_fewshot_navigate \
  bbh_cot_fewshot_object_counting \
  bbh_cot_fewshot_penguins_in_a_table \
  bbh_cot_fewshot_reasoning_about_colored_objects \
  bbh_cot_fewshot_ruin_names \
  bbh_cot_fewshot_salient_translation_error_detection \
  bbh_cot_fewshot_snarks \
  bbh_cot_fewshot_sports_understanding \
  bbh_cot_fewshot_temporal_sequences \
  bbh_cot_fewshot_tracking_shuffled_objects_five_objects \
  bbh_cot_fewshot_tracking_shuffled_objects_seven_objects \
  bbh_cot_fewshot_tracking_shuffled_objects_three_objects \
  bbh_cot_fewshot_web_of_lies \
  bbh_cot_fewshot_word_sorting
do
    # dsb baseline
    CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
        --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},block_length=${block_length},add_bos_token=true,alg=entropy,dsb=true,show_speed=True,outp_path=evals_results_${model_name}/dsb_baseline/${task}-ns0-${length}/results.jsonl \
        --tasks ${task} \
        --num_fewshot ${num_fewshot} \
        --batch_size 1 \
        --output_path evals_results_${model_name}/dsb_baseline/${task}-ns0-${length} --log_samples

    # dsb baseline + max_block_length
    CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
        --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},block_length=${block_length},add_bos_token=true,alg=entropy,dsb=true,max_block_length=${block_length},show_speed=True,outp_path=evals_results_${model_name}/dsb_baseline_32/${task}-ns0-${length}/results.jsonl \
        --tasks ${task} \
        --num_fewshot ${num_fewshot} \
        --batch_size 1 \
        --output_path evals_results_${model_name}/dsb_baseline_32/${task}-ns0-${length} --log_samples

    # dsb + parallel + max_block_length
    CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
        --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,dsb=true,max_block_length=${block_length},show_speed=True,outp_path=evals_results_${model_name}/dsb_parallel_32/${task}-ns0-${length}/results.jsonl \
        --tasks ${task} \
        --num_fewshot ${num_fewshot} \
        --batch_size 1 \
        --output_path evals_results_${model_name}/dsb_parallel_32/${task}-ns0-${length} --log_samples

    # dsb + cache + parallel + max_block_length
    CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
        --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,prefix_window=4,dsb=true,max_block_length=${block_length},use_cache=true,show_speed=True,outp_path=evals_results_${model_name}/dsb_cache_parallel_32/${task}-ns0-${length}/results.jsonl \
        --tasks ${task} \
        --num_fewshot ${num_fewshot} \
        --batch_size 1 \
        --output_path evals_results_${model_name}/dsb_cache_parallel_32/${task}-ns0-${length} --log_samples
done