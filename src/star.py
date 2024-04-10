import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba import MultiLayerMamba
# torch.autograd.set_detect_anomaly(True)


def gather_features(spatial_input_embedded, component_sizes, num_emb):
    max_peds = max(component_sizes)
    # 初始化结果张量
    N_of_scenes = len(component_sizes)
    gathered_embeddings = torch.zeros(N_of_scenes, max_peds, num_emb, device=spatial_input_embedded.device)
    
    # 填充结果张量
    start_idx = 0
    for i, size in enumerate(component_sizes):
        end_idx = start_idx + size
        gathered_embeddings[i, :size, :] = spatial_input_embedded[start_idx:end_idx]
        start_idx = end_idx
    
    return gathered_embeddings

def get_noise(shape, noise_type):
    if noise_type == "gaussian":
        return torch.randn(shape).cuda()
    elif noise_type == "uniform":
        return torch.rand(*shape).sub(0.5).mul(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

def slice_and_concat(emb_features, component_sizes):
    """
    高效地根据component_sizes对emb_features进行切片并拼接。
    
    参数:
    - emb_features: 形状为[scenes, MAX N of peds in individual scene, Embedded Feature]的Tensor。
    - component_sizes: 每个scene中有效pedestrians的数量列表。
    
    返回:
    - output_tensor: 形状为[sum(component_sizes), Embedded Feature]的新Tensor。
    """
    # 通过列表解析和torch.cat直接进行切片和拼接
    output_tensor = torch.cat([emb_features[i, :size, :] for i, size in enumerate(component_sizes)], dim=0)
    return output_tensor

class STAR(torch.nn.Module):

    def __init__(self, args, model_Hparameters):
        super(STAR, self).__init__()
        # model_Hparameters contains 3 Hyper-parameters:
        #     - n_layers: # of layers in the temporal encoder
        #     - ratio: The number of mambas used by the spatial layer is multiple times more than that used by the temporal layer
        #     - embedding
        # set parameters for network architecture
        self.emb = int(model_Hparameters["embedding"])
        self.temp_layers = int(model_Hparameters["n_layers"])
        self.spa_layers = int(self.temp_layers * model_Hparameters["ratio"])

        self.output_size = 2
        self.dropout_prob = model_Hparameters["dropout"]
        self.args = args

        self.spatial_encoder_1 = MultiLayerMamba(d_model=self.emb, n_layer = self.spa_layers, bi=True)
        self.spatial_encoder_2 = MultiLayerMamba(d_model=self.emb, n_layer = self.spa_layers, bi=True)
        # d_model = 64 for selective copying 
        self.temporal_encoder_1 = MultiLayerMamba(d_model=self.emb, n_layer = self.temp_layers, bi=False)
        self.temporal_encoder_2 = MultiLayerMamba(d_model=self.emb, n_layer = self.temp_layers, bi=False)

        # Linear layer to map input to embedding
        self.input_embedding_layer_temporal = nn.Linear(2, self.emb)
        self.input_embedding_layer_spatial = nn.Linear(2, self.emb)

        # Linear layer to output and fusion
        self.output_layer = nn.Linear(int(self.emb+16), 2)
        self.fusion_layer = nn.Linear(int(self.emb*2), self.emb)

        # ReLU and dropout init
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(self.dropout_prob)
        self.dropout_in2 = nn.Dropout(self.dropout_prob)

    def get_st_ed(self, batch_num):
        """

        :param batch_num: contains number of pedestrians in different scenes for a batch
        :type batch_num: list
        :return: st_ed: list of tuple contains start index and end index of pedestrians in different scenes
        :rtype: list
        """
        cumsum = torch.cumsum(batch_num, dim=0)
        st_ed = []
        for idx in range(1, cumsum.shape[0]):
            st_ed.append((int(cumsum[idx - 1]), int(cumsum[idx])))

        st_ed.insert(0, (0, int(cumsum[0])))

        return st_ed

    def get_node_index(self, seq_list):
        """

        :param seq_list: mask indicates whether pedestrain exists
        :type seq_list: numpy array [F, N], F: number of frames. N: Number of pedestrians (a mask to indicate whether
                                                                                            the pedestrian exists)
        :return: All the pedestrians who exist from the beginning to current frame
        :rtype: numpy array
        """
        for idx, framenum in enumerate(seq_list):

            if idx == 0:
                node_indices = framenum > 0
            else:
                node_indices *= (framenum > 0)

        return node_indices

    def update_batch_pednum(self, batch_pednum, ped_list):
        """

        :param batch_pednum: batch_num: contains number of pedestrians in different scenes for a batch
        :type list
        :param ped_list: mask indicates whether the pedestrian exists through the time window to current frame
        :type tensor
        :return: batch_pednum: contains number of pedestrians in different scenes for a batch after removing pedestrian who disappeared
        :rtype: list
        """
        updated_batch_pednum_ = copy.deepcopy(batch_pednum).cpu().numpy()
        updated_batch_pednum = copy.deepcopy(batch_pednum)

        cumsum = np.cumsum(updated_batch_pednum_)
        new_ped = copy.deepcopy(ped_list).cpu().numpy()

        for idx, num in enumerate(cumsum):
            num = int(num)
            if idx == 0:
                updated_batch_pednum[idx] = len(np.where(new_ped[0:num] == 1)[0])
            else:
                updated_batch_pednum[idx] = len(np.where(new_ped[int(cumsum[idx - 1]):num] == 1)[0])

        return updated_batch_pednum

    def mean_normalize_abs_input(self, node_abs, st_ed):
        """

        :param node_abs: Absolute coordinates of pedestrians
        :type Tensor
        :param st_ed: list of tuple indicates the indices of pedestrians belonging to the same scene
        :type List of tupule
        :return: node_abs: Normalized absolute coordinates of pedestrians
        :rtype: Tensor
        """
        node_abs = node_abs.permute(1, 0, 2)
        for st, ed in st_ed:
            mean_x = torch.mean(node_abs[st:ed, :, 0])
            mean_y = torch.mean(node_abs[st:ed, :, 1])

            node_abs[st:ed, :, 0] = (node_abs[st:ed, :, 0] - mean_x)
            node_abs[st:ed, :, 1] = (node_abs[st:ed, :, 1] - mean_y)

        return node_abs.permute(1, 0, 2)

    def forward(self, inputs, iftest=False):
        # Unpack inputs tuple into respective variables
        # nodes_abs, nodes_norm, shift_value, seq_list, nei_lists, nei_num, batch_pednum = inputs
        nodes_abs, nodes_norm, shift_value, seq_list, scenes, batch_pednum = inputs
        num_Ped = nodes_norm.shape[1]
        # initializing outputs and GM
        # bs: 19 as default
        # outputs  [bs, num of ped, 2]
        # GM       [bs, num of ped, 32 (embedding size)]
        outputs = torch.zeros(nodes_norm.shape[0], num_Ped, 2).cuda()
        GM = torch.zeros(nodes_norm.shape[0], num_Ped, self.emb).cuda()

        noise = get_noise((1, 16), 'gaussian')

        # Loop through each frame in the sequence except the last one
        # T = framenum + 1
        for framenum in range(self.args.seq_length - 1):

            if framenum >= self.args.obs_length and iftest:

                node_index = self.get_node_index(seq_list[:self.args.obs_length])
                updated_batch_pednum = self.update_batch_pednum(batch_pednum, node_index)
                st_ed = self.get_st_ed(updated_batch_pednum)

                nodes_current = outputs[self.args.obs_length - 1:framenum, node_index]
                nodes_current = torch.cat((nodes_norm[:self.args.obs_length, node_index], nodes_current))
                node_abs_base = nodes_abs[:self.args.obs_length, node_index]
                node_abs_pred = shift_value[self.args.obs_length:framenum + 1, node_index] + outputs[
                                                                                           self.args.obs_length - 1:framenum,
                                                                                           node_index]
                node_abs = torch.cat((node_abs_base, node_abs_pred), dim=0)
                # We normalize the absolute coordinates using the mean value in the same scene
                node_abs = self.mean_normalize_abs_input(node_abs, st_ed)

            else:
                node_index = self.get_node_index(seq_list[:framenum + 1])
                # netlist
                # --------------------------------------
                component_sizes = []
                # 遍历每个场景
                for start, end in scenes:
                    # 获取当前场景中所有行人的在场情况
                    current_scene_presence = node_index[start:end+1]
                    # 计算并存储当前场景中在场行人的数量
                    component_sizes.append(current_scene_presence.sum().item())
                # --------------------------------------
                updated_batch_pednum = self.update_batch_pednum(batch_pednum, node_index)
                st_ed = self.get_st_ed(updated_batch_pednum)
                nodes_current = nodes_norm[:framenum + 1, node_index]
                # We normalize the absolute coordinates using the mean value in the same scene
                node_abs = self.mean_normalize_abs_input(nodes_abs[:framenum + 1, node_index], st_ed)

            # Temporal Embedding ---------------------------------------------------
            # nodes_current [T, N of Ped, 2-coordinate] 
            # temporal_input_embedded [T, N of Ped, 32-Embedded coorinate] 
            emb_layer = self.input_embedding_layer_temporal(nodes_current)
            temporal_relu = torch.relu(emb_layer)
            temporal_input_embedded = self.dropout_in(temporal_relu)
            if framenum != 0:
                # Concat the input embedding with previous temporal output stored into GM(Graph Memory)  
                # temporal_input_embedded[:framenum] = GM[:framenum, node_index]
                # assume that the current seqence t is t=4, 
                # replace ALL the embedded feature before t by pervious output from temporal-2
                temporal_input_embedded_clone = temporal_input_embedded.clone()
                temporal_input_embedded_clone[:framenum] = GM[:framenum, node_index]
                temporal_input_embedded = temporal_input_embedded_clone
            
            # Sptial Embedding -----------------------------------------------------
            # spatial_embedded1 [N of Ped, Embedded Feature]
            emb_layer_spa = self.input_embedding_layer_spatial(node_abs)
            spa_input_relu = torch.relu(emb_layer_spa)
            spatial_embedded1 = self.dropout_in2(spa_input_relu)[-1]
            # First Spatial -------------------------------------------------------
            # gathered_features [N of scenes, N of peds in individual scene, Embedded Feature]
            gathered_features = gather_features(spatial_embedded1, component_sizes, self.emb)
            # output of spatial_encoder [N of scenes, N of peds in individual scene, Emb]
            spatial1 = self.spatial_encoder_1(gathered_features)
            # slice
            # spatial_input_embedded [N of Ped, Embedded Feature]
            spatial_input_embedded = slice_and_concat(spatial1, component_sizes)
            # First Temporal -------------------------------------------------------
            # input of temporal_encoder_1 [N of Ped, T, 32-Embedded coorinate]
            # output of temporal_encoder_1 [N of Ped, Embedded Feature]
            temporal_input_embedded_last = self.temporal_encoder_1(temporal_input_embedded.permute(1, 0, 2))[:, -1, :]
            temporal_input_embedded = temporal_input_embedded[:-1]

            # fusion [N of Ped, 2*Embedded Features]
            fusion_feat = torch.cat((temporal_input_embedded_last, spatial_input_embedded), dim=1)
            fusion_feat = self.fusion_layer(fusion_feat)

            # Second Spatial -------------------------------------------------
            # fusion_feat [N of Ped, Embedded Feature]
            # gathered_fusion [N of scenes, MAX peds in this scene, Embedded Feature]
            gathered_fusion = gather_features(fusion_feat, component_sizes, self.emb)
            spatial2 = self.spatial_encoder_2(gathered_fusion)
            # spatial_input_embedded [N of Ped, Embedded Feature]
            spatial_embedded2 = slice_and_concat(spatial2, component_sizes)
            spatial_embedded2 = torch.unsqueeze(spatial_embedded2, 0)
            # Second Temporal -------------------------------------------------
            # input of temporal_encoder_2 [(T-1) from GM + 1 from spatial = T, N of Ped, Embedded Features]
            temporal_input_embedded = torch.cat((temporal_input_embedded, spatial_embedded2), dim=0)
            # output of temporal_encoder_2 [N of Ped, Embedded Features]
            temporal_input_embedded = self.temporal_encoder_2(temporal_input_embedded.permute(1, 0, 2))[:, -1, :]
            # add noise
            noise_to_cat = noise.repeat(temporal_input_embedded.shape[0], 1)
            temporal_input_embedded_wnoise = torch.cat((temporal_input_embedded, noise_to_cat), dim=1)
            outputs_current = self.output_layer(temporal_input_embedded_wnoise)
            
            # Update the outputs and GM with the current frame's output
            outputs_clone = outputs.clone()
            outputs_clone[framenum, node_index] = outputs_current
            outputs = outputs_clone
            # GM[framenum, node_index] = temporal_input_embedded
            GM_clone = GM.clone()
            # 然后对GM_clone进行操作
            GM_clone[framenum, node_index] = temporal_input_embedded
            GM = GM_clone
        return outputs
