#!/bin/bash

export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=5

# Train the Global Object-Centric Learning moudel of GOLD for feature reconstruction
python -m torch.distributed.launch --nproc_per_node 1  --master_port 20510  --use-env train_ocl.py \
    --data_name  gso  \
    --data_path  ../dataset/gso.h5\
    --model_name  gold_ocl \
    --batch_size 8  --num_slots 8 --slot_size 128 --int_size 122 --ext_size 6 --bck_size 8\
    --seed 3496339326 --epochs 1000 --num_cls 10\
    --reg_start 0.001 --reg_final 0.001 --reg_steps 300000 \
    --kld_start 1 --kld_final 1 --kld_start_steps 0  --kld_final_steps 80000\
    --tau_start 1.0 --tau_final 0.1 --tau_start_steps 100000  --tau_final_steps 200000\
    # --checkpoint_path logs/gso/gold_ocl/checkpoint.pt.tar
    
# Train the Image Encoder-Decoder module for image reconstruction
python -m torch.distributed.launch --nproc_per_node 1  --master_port 20510 --use-env train_vqvae.py \
    --data_name  gso  \
    --data_path  ../dataset/gso.h5  \
    --ocl_ckp_path logs/gso/gold_ocl/best_model.pt \
    --model_name  gold_vqdec\
    --batch_size 4 --num_slots 8 --slot_size 128 --int_size 122 --ext_size 6 --bck_size 8\
    --epochs 1000  \
    --seed 3496339326 \
    # --checkpoint_path logs/gso/gold_vqdec/checkpoint.pt.tar

