#!/bin/bash

DATASET="$1"
MODEL="$2"
DEVICE="$3"
COMPLEX_TYPE="$4"
TASK="$5"

python -m tools.train_eval \
  --batch_size 128 \
  --complex_type "$COMPLEX_TYPE" \
  --conv_type B \
  --dataset "$DATASET" \
  --device "$DEVICE" \
  --drop_position lin2 \
  --drop_rate 0.4 \
  --emb_dim 16 \
  --epochs 200 \
  --eval_metric accuracy \
  --exp_name pcn-cosc \
  --final_readout sum \
  --folds 10 \
  --graph_norm bn \
  --init_method sum \
  --jump_mode None \
  --lr 0.001 \
  --lr_scheduler StepLR \
  --lr_scheduler_decay_rate 0.2 \
  --lr_scheduler_decay_steps 50 \
  --max_dim 2 \
  --model "$MODEL" \
  --nonlinearity relu \
  --num_fc_layers 2 \
  --num_layers 4 \
  --num_layers_combine 1 \
  --num_layers_update 1 \
  --preproc_jobs 32 \
  --readout sum \
  --task_type classification \
  --train_eval_period 50 \
  --task "$TASK"  \