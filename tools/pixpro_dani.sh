#!/bin/bash

set -e
set -x

data_dir="../../ssda/data/cityscapes/leftImg8bit_tiny"
output_dir="./output/pixpro_dani"

CUDA_VISIBLE_DEVICES=0 python \
    main_pretrain.py \
    --data-dir ${data_dir} \
    --output-dir ${output_dir} \
    \
    --zip --cache-mode no \
    --crop 0.08 \
    --aug BYOL \
    --dataset ImageNet \
    --batch-size 128 \
    \
    --model PixPro \
    --arch resnet50 \
    --head-type early_return \
    \
    --optimizer lars \
    --base-lr 1.0 \
    --weight-decay 1e-5 \
    --warmup-epoch 5 \
    --epochs 100 \
    \
    --save-freq 10 \
    --auto-resume \
    \
    --pixpro-p 2 \
    --pixpro-momentum 0.99 \
    --pixpro-pos-ratio 0.7 \
    --pixpro-transform-layer 1 \
    --pixpro-ins-loss-weight 0. \
