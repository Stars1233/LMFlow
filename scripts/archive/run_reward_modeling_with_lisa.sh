#!/bin/bash
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.
# Parses arguments
model_name_or_path=google/gemma-2b-it
train_dataset_path=data/ultrafeedback-binarized-preferences-cleaned/train
eval_dataset_path=data/ultrafeedback-binarized-preferences-cleaned/train
output_dir=output_models/reward_modeling_lisa
conversation_template=gemma
lisa_activated_layers=1
lisa_interval_steps=20

# Safety related arguments
trust_remote_code=0

while [[ $# -ge 1 ]]; do
  key="$1"
  case ${key} in
    -m|--model_name_or_path)
      model_name_or_path="$2"
      shift
      ;;
    --train_dataset_path)
      train_dataset_path="$2"
      shift
      ;;
    --eval_dataset_path)
      eval_dataset_path="$2"
      shift
      ;;
    --lisa_activated_layers)
      lisa_activated_layers="$2"
      shift
      ;;
    --lisa_interval_steps)
      lisa_interval_steps="$2"
      shift
      ;;
    -o|--output_model_path)
      output_dir="$2"
      shift
      ;;
    --conversation_template)
      conversation_template="$2"
      shift
      ;;
    --trust_remote_code)
      trust_remote_code="$2"
      shift
      ;;
    *)
      echo "error: unknown option \"${key}\"" 1>&2
      exit 1
  esac
  shift
done

# Finetune
exp_id=reward_modeling_lisa
project_dir=$(cd "$(dirname $0)"/..; pwd)
log_dir=${project_dir}/log/${exp_id}
mkdir -p ${output_dir} ${log_dir}

accelerate launch --config_file configs/accelerate_fsdp_config.yaml \
    examples/reward_modeling.py \
        --model_name_or_path ${model_name_or_path} \
        --arch_type "text_regression" \
        --do_train True \
        --dataset_path ${train_dataset_path} \
        --conversation_template ${conversation_template} \
        --output_dir ${output_dir} --overwrite_output_dir \
        --use_flash_attention True \
        --block_size 4096 \
        --learning_rate 1e-5 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --num_train_epochs 2 \
        --weight_decay 0.001 \
        --evaluation_strategy "steps" \
        --save_strategy "steps" \
        --save_steps 999999 \
        --gradient_accumulation_steps 32 \
        --gradient_checkpointing True \
        --remove_unused_columns False \
        --bf16 True \
        --torch_dtype bfloat16 \
        --logging_strategy "steps" \
        --logging_steps 10 \
        --optim "paged_adamw_32bit" \
        --lr_scheduler_type "cosine" \
        --warmup_ratio 0.03 \
        --report_to 'wandb' \
        --run_name ${exp_id} \
        --do_eval True \
        --eval_dataset_path ${eval_dataset_path} \
        --eval_steps 999999 \
        --use_lisa True \
        --lisa_activated_layers ${lisa_activated_layers} \
        --lisa_interval_steps ${lisa_interval_steps} \
        | tee ${log_dir}/train.log \
        2> ${log_dir}/train.err