#!/bin/bash

# 检查是否提供了足够的参数
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <learning_rate> <cuda_device>"
    exit 1
fi

# 读取命令行中提供的学习率和CUDA设备号
learning_rate=$1
cuda_device=$2

# 定义参数对
declare -a pairs=(
    "1 zara1"
)

# 循环遍历每对参数
for pair in "${pairs[@]}"; do
    # 读取X和Y值
    read X Y <<< "$pair"
    
    # 构建并执行训练命令，设置CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=${cuda_device} python trainval.py \
        --learning_rate ${learning_rate} \
        --determine ${X} \
        --num_epochs 100 \
        --patience 20\
        --test_set ${Y} \
        --emb 48  \
        --save_base_dir "result_hspa_lr_${learning_rate}_determine_${X}_test_set_${Y}" 
done


    # CUDA_VISIBLE_DEVICES=${cuda_device} python trainval.py \
    #     --learning_rate ${learning_rate} \
    #     --scheduler_method Cosine \
    #     --num_epochs 50 \
    #     --save_base_dir "hspa_${learning_rate}_${X}_${Y}" \
    #     --n_layers ${X} \
    #     --n_encoders ${Y}