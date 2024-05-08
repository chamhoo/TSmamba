import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import TSMambaBlock



def get_noise(shape, noise_type):
    if noise_type == "gaussian":
        return torch.randn(shape).cuda()
    elif noise_type == "uniform":
        return torch.rand(*shape).sub(0.5).mul(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


class TSModel(nn.Module):

    def __init__(self, emb=32, n_layers=3, n_encoders=1, config=None):
        super(TSModel, self).__init__()
        # Embedding
        self.input_1 = nn.Linear(2, 128)  # 第一个全连接层
        self.input_2 = nn.Linear(128, emb)  # 第二个全连接层，输出到嵌入维度
        # 创建多个TSMambaBlock层
        self.blocks = nn.ModuleList([TSMambaBlock(emb, n_encoders, **config) for _ in range(n_layers)])
        self.output = nn.Linear(emb, 2)
    
    def forward(self, inputs, gm, batch_pednum):

        # 将输入通过嵌入层
        inputs = F.relu(self.input_1(inputs))  # 应用ReLU激活函数
        inputs = self.input_2(inputs)  # 不在最后一个层使用激活函数

        # separarte x
        x_part1 = inputs[:-1]  
        x = inputs[-1]
        is_initial = True
        # 顺序通过每个TSMambaBlock
        for block in self.blocks:
            x = block(x, batch_pednum, gm, x_part1, is_initial)   # x, batch_pednum, previous_expression
            is_initial = False
        # add noise
        # noise = get_noise((1, 16), 'gaussian')
        # noise_to_cat = noise.repeat(x.shape[0], 1)
        # x_with_noise = torch.cat((x, noise_to_cat), dim=1)

        # output
        return self.output(x), x
    