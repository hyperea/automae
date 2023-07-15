#!/bin/bash
CHKPT="799"
IMAGENET_DIR='/data/ImageNet'
OUTPUT_DIR='/data/mae-finetuning'
PRETRAIN_CHKPT="/data/mae-pg/pretrain_scheduling_attnmapGANbmask0.5xgtopk800/checkpoint-$CHKPT.pth" 
LOG_DIR="./output_dir/finetuning_${CHKPT}"

OMP_NUM_THREADS=1 python -m torch.distributed.launch --master_port 24679 --nproc_per_node=8 main_finetune.py \
    --accum_iter 2 \
    --batch_size 64 \
    --model vit_base_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path ${IMAGENET_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${LOG_DIR} \
    --num_workers 12
