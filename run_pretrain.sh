#!/bin/bash
IMAGENET_DIR="/data/ImageNet"
OUTPUT_DIR='/data/checkpoints/pretrain_scheduling_attnmapGANbmask0.5xgtopk800'
LOG_DIR='./output_dir/pretrain_scheduling_attnmapGANbmask0.5xgtopk800'

OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 main_pretrain.py \
    --batch_size 64 \
    --accum_iter 8 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path ${IMAGENET_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${LOG_DIR} \
    --num_workers 12 \
