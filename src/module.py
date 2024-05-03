from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
import time


from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn


def gather_features(spatial_input_embedded, updated_batch_pednum, num_emb):
    max_peds = int(max(updated_batch_pednum).item())
    # 初始化结果张量
    N_of_scenes = len(updated_batch_pednum)
    gathered_embeddings = torch.zeros(N_of_scenes, max_peds, num_emb, device=spatial_input_embedded.device)
    
    # 填充结果张量
    start_idx = 0
    for i, size in enumerate(updated_batch_pednum):
        end_idx = start_idx + size
        gathered_embeddings[i, :size, :] = spatial_input_embedded[start_idx:end_idx]
        start_idx = end_idx
    
    return gathered_embeddings



def slice_and_concat(emb_features, updated_batch_pednum):
    """
    高效地根据updated_batch_pednum对emb_features进行切片并拼接。
    
    参数:
    - emb_features: 形状为[scenes, MAX N of peds in individual scene, Embedded Feature]的Tensor。
    - updated_batch_pednum: 每个scene中有效pedestrians的数量列表。
    
    返回:
    - output_tensor: 形状为[sum(updated_batch_pednum), Embedded Feature]的新Tensor。
    """
    # 通过列表解析和torch.cat直接进行切片和拼接
    output_tensor = torch.cat([emb_features[i, :size, :] for i, size in enumerate(updated_batch_pednum)], dim=0)
    return output_tensor




class TSMambaBlock(nn.Module):
    def __init__(self, emb, tmp_config, spa_config):
        super().__init__()
        self.emb = emb
        self.norm = nn.LayerNorm(normalized_shape=emb)

        self.tempblock = Mamba(emb, **tmp_config)
        self.spablock = Mamba(emb, **spa_config)

    def forward(self, x, batch_pednum, previous_expression):
        # norm
        x = self.norm(x.to(dtype=self.norm.weight.dtype))
        
        # tempblock
        combined_expression = torch.cat((previous_expression, x.unsqueeze(0)), dim=0)
        temp = combined_expression.permute(1, 0, 2)  # Reordering dimensions to [ped, 4, dim]
        temp = self.tempblock(temp)[:, -1, :]
        
        # spablock
        spa = gather_features(x, batch_pednum, self.emb)
        spa = self.spablock(spa)
        spa = slice_and_concat(spa, batch_pednum)

        return x + temp + spa