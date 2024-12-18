# models/dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler


class StockDataset(Dataset):
    def __init__(self, features, labels, seq_length, transform=True):
        """
        特征和标签数据集

        Parameters:
        features (DataFrame): 特征数据
        labels (Series): 标签数据
        seq_length (int): 序列长度
        transform (bool): 是否进行标准化
        """
        self.seq_length = seq_length

        if transform:
            self.scaler = StandardScaler()
            features = self.scaler.fit_transform(features)

        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features) - self.seq_length

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_length]
        y = self.labels[idx + self.seq_length]
        return x, y

    def get_scaler(self):
        """返回标准化器"""
        return self.scaler if hasattr(self, 'scaler') else None


class StockInferenceDataset(Dataset):
    def __init__(self, features, seq_length, scaler=None):
        """
        用于推理的数据集

        Parameters:
        features (DataFrame): 特征数据
        seq_length (int): 序列长度
        scaler (StandardScaler): 训练集的标准化器
        """
        self.seq_length = seq_length

        if scaler is not None:
            features = scaler.transform(features)

        self.features = torch.FloatTensor(features)

    def __len__(self):
        return len(self.features) - self.seq_length + 1

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_length]
        return x