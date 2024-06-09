import torch
import torch.nn as nn

# # 初始化模型
# model = TimeSeriesTransformer(embedding_size, num_layers, num_heads)

# # 调整输入以匹配Transformer的期望维度: [seq_length, batch_size, embedding_size]
# input_tensor = input_tensor.permute(1, 0, 2)

# print("输出维度:", output.shape)  # [seq_length, batch_size, embedding_size]

# 创建模型
class TimeSeriesTransformer(nn.Module):
    def __init__(self, embedding_size=48, num_layers=1, num_heads=8, batch_first=True):
        super(TimeSeriesTransformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=num_heads,
            dim_feedforward=2048,  # Feed forward网络的维度
            dropout=0.1,            # Dropout概率
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, src):
        # src的维度应该是 [seq_length, batch_size, embedding_size]
        output = self.transformer_encoder(src)
        return output



# # 初始化模型
# model = LSTMTimeSeriesModel(embedding_size, hidden_size, num_layers)

# 创建LSTM模型
class LSTMTimeSeriesModel(nn.Module):
    def __init__(self, input_size=48, hidden_size=48, num_layers=1):
        super(LSTMTimeSeriesModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)  # 将输出维度调整回 embedding_size

    def forward(self, x):
        # x的维度为 [batch_size, seq_length, embedding_size]
        lstm_out, (h_n, c_n) = self.lstm(x)
        # 通过线性层调整维度
        output = self.linear(lstm_out)
        return output

