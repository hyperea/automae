#!/bin/bash
CHKPT="399"
IMAGENET_DIR="/data/ImageNet"
OUTPUT_DIR='/data/mae-linprobing'
PRETRAIN_CHKPT="/data/checkpoints/pretrain_scheduling_attnmapGANbmask0.5xgtopk800/checkpoint-$CHKPT.pth" 
LOG_DIR="./output_dir/linprobe_${CHKPT}_scheduling_attnmapGANbmask0.5xgtopk800"

OMP_NUM_THREADS=1 torchrun --master_port 24678 --nproc_per_node=8 main_linprobe.py \
    --batch_size 512 \
    --accum_iter 4 \
    --model vit_base_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --cls_token \
    --epochs 90 \
    --warmup_epochs 10 \
    --blr 0.1 \
    --weight_decay 0.0 \
    --data_path ${IMAGENET_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${LOG_DIR} \
    --num_workers 8
