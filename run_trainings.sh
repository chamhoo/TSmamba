#!/bin/bash

# 检查是否提供了足够的参数
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <cuda_device>"
    exit 1
fi

# 读取命令行中提供的学习率和CUDA设备号
cuda_device=$1

# 定义参数对
declare -a pairs=(
    "eth"
    "hotel"
    "zara1"
    "zara2"
    "univ"
)
# 循环遍历每对参数
for pair in "${pairs[@]}"; do
    # 读取X和Y值
    read X <<< "$pair"
    
    # 构建并执行训练命令，设置CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=${cuda_device} python trainval.py \
        --learning_rate 0.001 \
        --determine 1 \
        --num_epochs 200 \
        --patience 20\
        --test_set ${X} \
        --emb 48  \
        --save_base_dir "result_puretemp_testset_${X}"  
done
