# models/lstm_model.py

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        """
        LSTM模型定义

        Parameters:
        input_dim (int): 输入特征维度
        hidden_dim (int): 隐藏层维度
        num_layers (int): LSTM层数
        output_dim (int): 输出维度
        dropout (float): dropout率
        """
        super(LSTMModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # LSTM层
        out, _ = self.lstm(x, (h0, c0))

        # 只使用最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out

    def predict(self, x):
        """用于推理的方法"""
        self.eval()
        with torch.no_grad():
            out = self.forward(x)
            return torch.softmax(out, dim=1)