#!/bin/bash

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"


module load cuda/12.2.0

retain_split="RTL_QA_retain"
holdout_split="RTL_QA_holdout"
base_model="Llama-3.1-8B-Instruct"

retain_setting="RTL_Coder"
forget_setting="RTL_VerilogEval" 
holdout_setting="RTLLM"
batch_size=4
trainer="GradAscent" # here could be replace by any trainer like GradDiff, NPO, SimNPO, RMU, etc.
unlearn_epoch=2 # unlearn epoch

# Prepare dataset once
python src/data_process.py ${retain_setting} ${forget_setting} ${holdout_setting}

# learned_model="saves/finetune/out_VerilogEval_final"

for top_p in 0 0.25 0.5 0.75 1.0
do
  for temperature in 0.2 0.4 0.6 0.8 1.0
  do
    echo "Running with top_p=${top_p}, temperature=${temperature}"
    
    folder_name="eval_unlearn_top_p_${top_p}_temp_${temperature}"
    task_name="RTL_VerilogEval_Unlearn_${trainer}_ep${unlearn_epoch}"
    unlearn_model="saves/unlearn/RTL_VerilogEval_Unlearn_${trainer}_ep${unlearn_epoch}"

    CUDA_VISIBLE_DEVICES=0 python src/eval.py \
      experiment=eval/tofu/RTL_default.yaml \
      forget_split=${forget_split} \
      holdout_split=${holdout_split} \
      model=${base_model} \
      task_name=${task_name} \
      model.model_args.pretrained_model_name_or_path=${unlearn_model} \
      paths.output_dir=saves/unlearn/${task_name}/${folder_name} \
      retain_logs_path=saves/unlearn/${task_name}/${folder_name}/TOFU_EVAL.json \
      eval.tofu.metrics.forget_Q_A_ROUGE.batch_size=128 \
      eval.tofu.metrics.forget_Q_A_Prob.batch_size=${batch_size} \
      eval.tofu.metrics.privleak.pre_compute.mia_min_k.batch_size=${batch_size} \
      eval.tofu.metrics.mia_min_k_plus_plus.batch_size=${batch_size} \
      eval.tofu.metrics.mia_min_k.batch_size=${batch_size} \
      eval.tofu.metrics.forget_Q_A_ROUGE.generation_args.top_p=${top_p} \
      eval.tofu.metrics.forget_Q_A_ROUGE.generation_args.do_sample=true \
      eval.tofu.metrics.forget_Q_A_ROUGE.generation_args.temperature=${temperature} \
  
  done
done