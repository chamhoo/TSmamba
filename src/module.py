from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
import time


from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn

from .encoders import TimeSeriesTransformer, LSTMTimeSeriesModel


def gather_features(spatial_input_embedded, updated_batch_pednum, num_emb):
    N_of_scenes = updated_batch_pednum.shape[0]
    max_peds = torch.max(updated_batch_pednum)

    gathered_embeddings = torch.zeros(
        N_of_scenes*max_peds, 
        num_emb, 
        device=spatial_input_embedded.device)
    
    idx_tensor = torch.arange(
        start=0, 
        end=len(spatial_input_embedded), 
        device=spatial_input_embedded.device)
    
    residual_idx = torch.cat((torch.tensor([0], device=spatial_input_embedded.device), torch.cumsum(max_peds-updated_batch_pednum, dim=0)[:-1]))
    scene_idx = torch.repeat_interleave(residual_idx, updated_batch_pednum)

    idx = scene_idx + idx_tensor
    
    gathered_embeddings.scatter_(0, idx[:, None].expand(-1, num_emb), spatial_input_embedded)
    
    return gathered_embeddings.view(N_of_scenes, max_peds, num_emb)



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
    indices = torch.arange(emb_features.size(1), device=emb_features.device).repeat(emb_features.size(0), 1)
    mask = indices < updated_batch_pednum[:, None]
    return emb_features[mask].view(-1, emb_features.size(2))




class TSMambaBlock(nn.Module):
    def __init__(self, emb, args, tmp_config, pre_config, spa_config):
        super().__init__()
        self.emb = emb
        self.args = args
        self.norm = nn.LayerNorm(normalized_shape=emb)
        self.earlyfuse = nn.Linear(2*emb, emb)
        if self.args.v == 1:
            self.tempblock = Mamba(emb, **tmp_config)
        elif self.args.v in [3, 5]:
            # embedding_size=48, num_layers=1, num_heads=8
            self.tempblock = TimeSeriesTransformer(emb, 1, 8)
        elif self.args.v in [2, 4]:
            # input_size=48, hidden_size=48, num_layers=1
            self.tempblock = LSTMTimeSeriesModel(emb, emb, 1)

    def forward(self, x, gm, batch_pednum, pre_traj):
        # norm
        x = self.norm(x.to(dtype=self.norm.weight.dtype))
        
        combined_expression = torch.cat((pre_traj, x.unsqueeze(0)), dim=0)
        temp = combined_expression.permute(1, 0, 2)  # Reordering dimensions to [ped, 4, dim]
        temp = self.tempblock(temp)[:, -1, :]

        # prefuse
        # if gm.shape[0] != 0:
        #     fuse = gm.permute(1, 0, 2)
        #     fuse = self.preblock(fuse)
        #     spa = self.earlyfuse(torch.concat((x, fuse[:,-1,:]), dim=1))
        # else:
        #     spa = x
        # spa = self.earlyfuse(torch.concat((x, temp), dim=1))
        # # spa
        # spa = gather_features(spa, batch_pednum, self.emb)
        # spa = self.spablock(spa)
        # spa = slice_and_concat(spa, batch_pednum)
        return x + temp #+ spa