#!/bin/bash

# 检查是否提供了足够的参数
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <cuda_device>"
    exit 1
fi

# 读取命令行中提供的学习率和CUDA设备号
version=$1
cuda_device=$2


    
# 构建并执行训练命令，设置CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=${cuda_device} python trainval.py \
    --learning_rate 0.001 \
    --determine 1 \
    --num_epochs 200 \
    --patience 20\
    --test_set "zara1" \
    --emb 48  \
    --v ${version}
    --save_base_dir "result_encoders_testset_zara1_version_${version}"  
