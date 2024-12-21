# lstm_model.py
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 3)  # 3分类

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


class LSTMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim=None, hidden_dim=64, num_layers=2, sequence_length=10):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.model = None

    def fit(self, X, y):
        if self.input_dim is None:
            self.input_dim = X.shape[1]

        self.model = LSTMModel(self.input_dim, self.hidden_dim, self.num_layers)
        dataset = self._prepare_sequences(X, y)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters())

        for epoch in range(10):  # 训练10个epoch
            self.model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not fitted yet!")

        self.model.eval()
        dataset = self._prepare_sequences(X, np.zeros(len(X)))
        loader = DataLoader(dataset, batch_size=32)

        predictions = []
        with torch.no_grad():
            for batch_X, _ in loader:
                outputs = self.model(batch_X)
                _, pred = torch.max(outputs, 1)
                predictions.extend(pred.numpy())

        return np.array(predictions)

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model not fitted yet!")

        self.model.eval()
        dataset = self._prepare_sequences(X, np.zeros(len(X)))
        loader = DataLoader(dataset, batch_size=32)

        probabilities = []
        with torch.no_grad():
            for batch_X, _ in loader:
                outputs = self.model(batch_X)
                probs = torch.softmax(outputs, dim=1)
                probabilities.extend(probs.numpy())

        return np.array(probabilities)

    def _prepare_sequences(self, X, y):
        sequences = []
        targets = []
        for i in range(len(X) - self.sequence_length):
            seq = X[i:i + self.sequence_length]
            target = y[i + self.sequence_length]
            sequences.append(seq)
            targets.append(target)
        return TensorDataset(
            torch.FloatTensor(sequences),
            torch.LongTensor(targets)
        )