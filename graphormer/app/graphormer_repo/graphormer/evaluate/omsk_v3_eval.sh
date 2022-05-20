#!/usr/bin/env bash

    
python evaluate_single.py \
    --user-dir ../../graphormer \
    --num-workers 10 \
    --ddp-backend=legacy_ddp \
    --dataset_name omsk \
    --dataset-source pyg \
    --num-atoms 6656 \
    --task graph_prediction \
    --criterion l1_loss \
    --arch graphormer_slim \
    --num-classes 1 \
    --batch-size 8 \
    --save-dir ../../examples/georides/omsk/ckpts/ \
    --split valid \
    --metric rmse 



