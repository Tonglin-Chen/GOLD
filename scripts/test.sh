#!/bin/bash

export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=6


python  test_ocl.py \
    --data_name  gso  \
    --data_path  ../dataset/gso.h5  \
    --model_name  gold_ocl \
    --ocl_ckp_path logs/gso/gold_ocl/best_model.pt \
    --batch_size 4  --num_slots 8 --slot_size 128 --int_size 122 --ext_size 6 --bck_size 8\
    --seed 3496339326 \


python  test_gold.py \
    --data_name  gso  \
    --data_path  ../dataset/gso.h5  \
    --model_name  gold \
    --ocl_ckp_path logs/gso/gold_ocl/best_model.pt \
    --gold_ckp_path logs/gso/gold_vqdec/best_model.pt \
    --batch_size 4  --num_slots 8 --slot_size 128 --int_size 122 --ext_size 6 --bck_size 8\
    --seed 3496339326 \
