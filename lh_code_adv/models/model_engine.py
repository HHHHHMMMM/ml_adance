# models/model_engine.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from ..config.config import Config
from .lstm_model import LSTMModel
from .dataset import StockDataset, StockInferenceDataset
import logging


class ModelEngine:
    def __init__(self, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None

    def train(self, train_data, valid_data, input_dim):
        """
        训练模型

        Parameters:
        train_data (StockDataset): 训练数据集
        valid_data (StockDataset): 验证数据集
        input_dim (int): 输入特征维度
        """
        # 保存标准化器
        self.scaler = train_data.get_scaler()

        # 创建数据加载器
        train_loader = DataLoader(
            train_data,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=2
        )

        valid_loader = DataLoader(
            valid_data,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=2
        )

        # 初始化模型
        self.model = LSTMModel(
            input_dim=input_dim,
            hidden_dim=Config.HIDDEN_DIM,
            num_layers=Config.NUM_LAYERS,
            output_dim=3  # 三分类：买入、卖出、持有
        ).to(self.device)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters())

        # 早停机制
        best_valid_loss = float('inf')
        patience = 5
        patience_counter = 0

        # 训练循环
        for epoch in range(Config.EPOCHS):
            # 训练阶段
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

            # 验证阶段
            self.model.eval()
            valid_loss = 0
            valid_correct = 0
            valid_total = 0

            with torch.no_grad():
                for batch_x, batch_y in valid_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)

                    valid_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    valid_total += batch_y.size(0)
                    valid_correct += (predicted == batch_y).sum().item()

            # 计算准确率
            train_accuracy = 100 * train_correct / train_total
            valid_accuracy = 100 * valid_correct / valid_total

            # 打印训练信息
            logging.info(f'Epoch [{epoch + 1}/{Config.EPOCHS}]')
            logging.info(f'Train Loss: {train_loss / len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%')
            logging.info(f'Valid Loss: {valid_loss / len(valid_loader):.4f}, Accuracy: {valid_accuracy:.2f}%')

            # 早停检查
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info("Early stopping triggered")
                    break

    def predict(self, features):
        """
        模型预测

        Parameters:
        features (DataFrame): 特征数据

        Returns:
        numpy.ndarray: 预测结果
        """
        if self.model is None:
            raise Exception("Model not trained yet!")

        # 创建推理数据集
        dataset = StockInferenceDataset(
            features,
            Config.SEQUENCE_LENGTH,
            self.scaler
        )

        dataloader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False
        )

        predictions = []
        self.model.eval()

        with torch.no_grad():
            for batch_x in dataloader:
                batch_x = batch_x.to(self.device)
                outputs = self.model.predict(batch_x)
                predictions.append(outputs.cpu().numpy())

        return np.concatenate(predictions)

    def load_model(self, model_path):
        """加载已训练的模型"""
        if self.model is None:
            raise Exception("Model not initialized!")
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()