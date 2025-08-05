#!/bin/bash

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

# Shared parameters
retain_split="RTL_QA_retain"
holdout_split="RTL_QA_holdout"
base_model="Llama-3.1-8B-Instruct"
finetuned_model="/saved/finetune/out_verilogeval_final"
per_device_train_batch_size=2 
gradient_accumulation_steps=2
unlearn_epoch=2

retain_setting="RTL_Coder"
forget_setting="RTL_VerilogEval"
holdout_setting="RTLLM"

# Prepare dataset once
python src/data_process.py ${retain_setting} ${forget_setting} ${holdout_setting}

# Trainer loop
for trainer in GradAscent #GradDiff DPO NPO SimNPO RMU
do
  task_name="RTL_VerilogEval_Unlearn_${trainer}_ep${unlearn_epoch}"

  if [ "$trainer" = "DPO" ]; then
    experiment="unlearn/tofu/RTL_idk.yaml"
    forget_split="RTL_QA_forget_idk"
  else
    experiment="unlearn/tofu/RTL_default.yaml"
    forget_split="RTL_QA_forget"
  fi

  echo "Starting unlearning with trainer: ${trainer}"

  CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
    src/train.py --config-name=salad.yaml \
    experiment=${experiment} \
    trainer=${trainer} \
    model=${base_model} \
    task_name=${task_name} \
    forget_split=${forget_split} \
    retain_split=${retain_split} \
    holdout_split=${holdout_split} \
    model.model_args.pretrained_model_name_or_path=${finetuned_model} \
    trainer.args.num_train_epochs=${unlearn_epoch} \
    retain_logs_path=saves/${task_name}/eval/${forget_setting}/TOFU_EVAL.json \
    trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
    trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
    trainer.args.ddp_find_unused_parameters=true \
    trainer.args.gradient_checkpointing=true

done
