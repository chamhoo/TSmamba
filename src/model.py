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

    def __init__(self, isdetermine, emb=32, config=None):
        super(TSModel, self).__init__()
        self.isdetermine = isdetermine
        # Embedding
        self.input_1 = nn.Linear(2, 128)  # 第一个全连接层
        self.input_2 = nn.Linear(128, emb)  # 第二个全连接层，输出到嵌入维度
        # self.earlyfuse = nn.Linear(2*emb, emb)
        # 创建多个TSMambaBlock层
        self.block = TSMambaBlock(emb, **config)
        if self.isdetermine:
            self.output = nn.Linear(emb, 2)
        else:
            self.output = nn.Linear(emb+16, 2)
        self.act = torch.nn.SiLU()
    
    def forward(self, inputs, gm, batch_pednum):

        # Inputs and embedding
        inputs = F.relu(self.input_1(inputs))  # 应用ReLU激活函数
        inputs = self.input_2(inputs)  # 不在最后一个层使用激活函数

        # separarte x
        x_part1 = inputs[:-1]  
        x = inputs[-1]

        # 顺序通过每个TSMambaBlock
        x = self.block(x, gm, batch_pednum, x_part1)   # x, batch_pednum, previous_expression
        if self.isdetermine:
            # add act
            return self.output(self.act(x)), x
        else:
            # add noise
            noise = get_noise((1, 16), 'gaussian')
            noise_to_cat = noise.repeat(x.shape[0], 1)
            x_with_noise = torch.cat((x, noise_to_cat), dim=1)
            # output
            return self.output(x_with_noise), x

    