import torch
import torch.nn as nn
from .model import TSModel



class PFA(torch.nn.Module):
    # Progressive Frame Accumulation Structure
    def __init__(self, args, model_Hparameters):
        super(PFA, self).__init__()
        self.args = args
        isdetermine = self.args.determine
        self.model = TSModel(isdetermine, args, **model_Hparameters)  # def __init__(self, emb=32, n_layers=3, config=None, previous_use_GM=True):
        self.emb = model_Hparameters["emb"]


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


    @staticmethod
    def get_node_index(seq_list):
        """
        Determine the indices of pedestrians who exist in all frames from the beginning up to each current frame using PyTorch.

        :param seq_list: A 2D PyTorch tensor where each element indicates the presence (values > 0) or absence (values == 0)
                        of a pedestrian in a frame. Shape: [F, N], where F is the number of frames and N is the number of pedestrians.
        :return: A boolean tensor indicating pedestrians who exist in every frame up to the current one.
        :rtype: torch.BoolTensor
        """
        # Convert the presence indicators to boolean (True if value > 0)
        presence = seq_list > 0
        # Compute a logical AND across the first dimension (frames)
        return torch.all(presence, dim=0)


    @staticmethod
    def update_batch_pednum(pednum, ped_list):
        """
        更新每个场景中的行人数量。
        :param pednum: 含有不同场景行人数量的列表
        :type list
        :param ped_list: 表示时间窗口内行人存在情况的张量
        :type tensor
        :return: 更新后的行人数量列表
        :rtype: list
        """
        cumsum = torch.cumsum(pednum.int(), dim=0)
        
        # 生成每个场景结束索引的掩码
        max_idx = cumsum[-1]
        idx_range = torch.arange(max_idx, device=pednum.device).unsqueeze(0)
        ends = cumsum.unsqueeze(1)
        starts = torch.cat((torch.tensor([0], device=pednum.device), cumsum[:-1])).unsqueeze(1)

        # 使用广播生成掩码
        mask = (idx_range >= starts) & (idx_range < ends)

        # 应用掩码并计算结果
        selected_ped_list = mask * ped_list[:max_idx].unsqueeze(0)
        updated_batch_pednum = selected_ped_list.sum(dim=1)
        return updated_batch_pednum.int()


    @staticmethod
    def mean_normalize_abs_input(node_abs, st_ed):
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


    def center_alignment_spa(self, data, st_ed):
        """
        :param data: Absolute coordinates of pedestrians
        :type Tensor
        :param st_ed: list of tuple indicates the indices of pedestrians belonging to the same scene
        :type List of tupule
        :return: data: Normalized absolute coordinates of pedestrians
        :rtype: Tensor
        """
        data = data.permute(1, 0, 2)
        for st, ed in st_ed:
            center = torch.mean(data[st:ed], axis=0)
            data[st:ed]  = data[st:ed] - center  # torch.Size([12, 1, 2])
        return data.permute(1, 0, 2)
    


    def forward(self, inputs, iftest=False):
        # Unpack inputs tuple into respective variables
        # nodes_abs, nodes_norm, shift_value, seq_list, nei_lists, nei_num, pednum = inputs
        nodes_abs, nodes_norm, shift_value, seq_list, scenes, pednum = inputs
        num_Ped = nodes_norm.shape[1]
        # initializing outputs and GM
        # bs: 19 as default
        # outputs  [bs, num of ped, 2]
        # GM       [bs, num of ped, 32 (embedding size)]
        outputs = torch.zeros(nodes_norm.shape[0], num_Ped, 2).cuda()
        GM = torch.zeros(nodes_norm.shape[0], num_Ped, self.emb).cuda()

        # Loop through each frame in the sequence except the last one
        # T = framenum + 1
        for framenum in range(self.args.seq_length - 1):

            if framenum >= self.args.obs_length and iftest:

                node_index = self.get_node_index(seq_list[:self.args.obs_length])
                batch_pednum = self.update_batch_pednum(pednum, node_index)

                # 对于 obs_length 之后的frames，在预测阶段将不能使用真实轨迹数据
                coord_pred = outputs[self.args.obs_length - 1:framenum, node_index]
                # temp
                coord_temp = torch.cat((nodes_norm[:self.args.obs_length, node_index], coord_pred))
                # spa
                coord_spa_pred = coord_pred + shift_value[self.args.obs_length:framenum + 1, node_index]
                coord_spa = torch.cat((nodes_abs[:self.args.obs_length, node_index], coord_spa_pred))


            else:
                node_index = self.get_node_index(seq_list[:framenum + 1])
                batch_pednum = self.update_batch_pednum(pednum, node_index)
                # temp
                coord_temp = nodes_norm[:framenum + 1, node_index]
                # spa
                coord_spa = nodes_abs[:framenum + 1, node_index]

            # norm spa
            st_ed = self.get_st_ed(pednum)
            coord_spa = self.center_alignment_spa(coord_spa, st_ed)

            if self.args.v in [2,3]:
                coord = coord_temp
            elif self.args.v in [1,4,5]:
                coord = coord_spa
            
            # forward
            output, output_emb = self.model(coord, GM[:framenum, node_index], batch_pednum)  # forward(self, x, gm, batch_pednum)
            # Update the outputs and GM with the current frame's output
            outputs[framenum, node_index] = output
            GM[framenum, node_index] = output_emb
        return outputs