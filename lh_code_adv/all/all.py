# backtesting/backtest_engine.py

import pandas as pd
import numpy as np
from ..config.config import Config


class BacktestEngine:
    def __init__(self, initial_capital=1000000):
        self.initial_capital = initial_capital
        self.positions = {}  # 持仓情况
        self.cash = initial_capital  # 可用现金
        self.trades = []  # 交易记录
        self.daily_portfolio_value = []  # 每日组合价值

    def run_backtest(self, signals_df, price_data):
        """
        运行回测

        Parameters:
        signals_df (DataFrame): 包含交易信号的DataFrame
        price_data (DataFrame): 价格数据

        Returns:
        dict: 回测结果
        """
        # 重置回测状态
        self.positions = {}
        self.cash = self.initial_capital
        self.trades = []
        self.daily_portfolio_value = []

        # 按日期排序
        dates = sorted(signals_df.index.unique())

        # 遍历每个交易日
        for date in dates:
            # 获取当日信号
            daily_signals = signals_df.loc[date]

            # 更新持仓价值
            self._update_portfolio_value(date, price_data)

            # 执行交易
            for symbol, signal in daily_signals.items():
                if signal != 0:  # 有交易信号
                    self._execute_trade(date, symbol, signal, price_data)

        # 计算回测结果
        return self._calculate_metrics()

    def _execute_trade(self, date, symbol, signal, price_data):
        """执行交易"""
        current_price = price_data.loc[date, symbol]['close']

        if signal > 0:  # 买入信号
            # 计算可买入数量
            available_money = self.cash * Config.POSITION_LIMIT
            shares = int(available_money / current_price / 100) * 100  # 买入数量向下取整到100的倍数

            if shares > 0:
                cost = shares * current_price * (1 + Config.COMMISSION_RATE)
                if cost <= self.cash:
                    self.positions[symbol] = self.positions.get(symbol, 0) + shares
                    self.cash -= cost
                    self.trades.append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'buy',
                        'shares': shares,
                        'price': current_price,
                        'cost': cost
                    })

        elif signal < 0:  # 卖出信号
            if symbol in self.positions and self.positions[symbol] > 0:
                shares = self.positions[symbol]
                revenue = shares * current_price * (1 - Config.COMMISSION_RATE)
                self.positions[symbol] = 0
                self.cash += revenue
                self.trades.append({
                    'date': date,
                    'symbol': symbol,
                    'action': 'sell',
                    'shares': shares,
                    'price': current_price,
                    'revenue': revenue
                })

    def _update_portfolio_value(self, date, price_data):
        """更新组合价值"""
        portfolio_value = self.cash

        # 计算持仓市值
        for symbol, shares in self.positions.items():
            if shares > 0:
                current_price = price_data.loc[date, symbol]['close']
                portfolio_value += shares * current_price

        self.daily_portfolio_value.append({
            'date': date,
            'value': portfolio_value
        })

    def _calculate_metrics(self):
        """计算回测指标"""
        df = pd.DataFrame(self.daily_portfolio_value)
        df.set_index('date', inplace=True)

        # 计算收益率
        df['returns'] = df['value'].pct_change()

        # 计算累计收益率
        total_return = (df['value'].iloc[-1] - self.initial_capital) / self.initial_capital

        # 计算年化收益率
        days = (df.index[-1] - df.index[0]).days
        annual_return = (1 + total_return) ** (365 / days) - 1

        # 计算夏普比率
        risk_free_rate = 0.03  # 假设无风险利率为3%
        excess_returns = df['returns'] - risk_free_rate / 365
        sharpe_ratio = np.sqrt(365) * excess_returns.mean() / excess_returns.std()

        # 计算最大回撤
        df['cummax'] = df['value'].cummax()
        df['drawdown'] = (df['cummax'] - df['value']) / df['cummax']
        max_drawdown = df['drawdown'].max()

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades),
            'final_value': df['value'].iloc[-1],
            'daily_returns': df['returns']
        }

# backtesting/visualization.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class BacktestVisualizer:
    def __init__(self):
        self.style_config = {
            'figure.figsize': (15, 8),
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10
        }

        plt.style.use('seaborn')
        for key, value in self.style_config.items():
            plt.rcParams[key] = value

    def plot_equity_curve(self, portfolio_values, benchmark_values=None):
        """绘制权益曲线"""
        plt.figure(figsize=(15, 8))

        # 绘制策略曲线
        plt.plot(portfolio_values.index, portfolio_values.values,
                 label='Strategy', linewidth=2)

        # 如果有基准数据，添加基准曲线
        if benchmark_values is not None:
            plt.plot(benchmark_values.index, benchmark_values.values,
                     label='Benchmark', linewidth=2, alpha=0.7)

        plt.title('Portfolio Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.grid(True)

        plt.savefig('equity_curve.png')
        plt.close()

    def plot_drawdown(self, portfolio_values):
        """绘制回撤图"""
        # 计算回撤
        rolling_max = portfolio_values.cummax()
        drawdowns = (portfolio_values - rolling_max) / rolling_max

        plt.figure(figsize=(15, 8))
        plt.plot(drawdowns.index, drawdowns.values, 'r', label='Drawdown')
        plt.fill_between(drawdowns.index, drawdowns.values, 0, alpha=0.3, color='red')

        plt.title('Portfolio Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.legend()
        plt.grid(True)

        plt.savefig('drawdown.png')
        plt.close()

    def plot_monthly_returns(self, returns):
        """绘制月度收益热力图"""
        # 计算月度收益
        monthly_returns = returns.resample('M').agg(lambda x: (1 + x).prod() - 1)
        monthly_returns_table = monthly_returns.groupby([
            monthly_returns.index.year,
            monthly_returns.index.month
        ]).first().unstack()

        plt.figure(figsize=(15, 8))
        sns.heatmap(monthly_returns_table, annot=True, fmt='.2%', cmap='RdYlGn')

        plt.title('Monthly Returns Heatmap')
        plt.xlabel('Month')
        plt.ylabel('Year')

        plt.savefig('monthly_returns.png')
        plt.close()

    def create_interactive_chart(self, price_data, signals, portfolio_values):
        """创建交互式图表"""
        fig = make_subplots(rows=2, cols=1, shared_xaxis=True,
                            vertical_spacing=0.03,
                            subplot_titles=('Price and Signals', 'Portfolio Value'))

        # 添加价格数据
        fig.add_trace(
            go.Candlestick(
                x=price_data.index,
                open=price_data['open'],
                high=price_data['high'],
                low=price_data['low'],
                close=price_data['close'],
                name='Price'
            ),
            row=1, col=1
        )

        # 添加买入信号
        buy_signals = signals[signals > 0]
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=price_data.loc[buy_signals.index, 'close'],
                mode='markers',
                marker=dict(symbol='triangle-up', size=10, color='green'),
                name='Buy Signal'
            ),
            row=1, col=1
        )

        # 添加卖出信号
        sell_signals = signals[signals < 0]
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=price_data.loc[sell_signals.index, 'close'],
                mode='markers',
                marker=dict(symbol='triangle-down', size=10, color='red'),
                name='Sell Signal'
            ),
            row=1, col=1
        )

        # 添加组合价值
        fig.add_trace(
            go.Scatter(
                x=portfolio_values.index,
                y=portfolio_values.values,
                name='Portfolio Value'
            ),
            row=2, col=1
        )

        # 更新布局
        fig.update_layout(
            title='Backtest Results',
            xaxis_title='Date',
            yaxis_title='Price',
            yaxis2_title='Portfolio Value',
            height=800
        )

        # 保存为HTML文件
        fig.write_html('interactive_backtest.html')

    def generate_report(self, backtest_results):
        """生成回测报告"""
        report = pd.DataFrame({
            'Metric': [
                'Total Return',
                'Annual Return',
                'Sharpe Ratio',
                'Max Drawdown',
                'Win Rate',
                'Average Win',
                'Average Loss',
                'Profit Factor'
            ],
            'Value': [
                f"{backtest_results['total_return']:.2%}",
                f"{backtest_results['annual_return']:.2%}",
                f"{backtest_results['sharpe_ratio']:.2f}",
                f"{backtest_results['max_drawdown']:.2%}",
                f"{backtest_results['win_rate']:.2%}",
                f"{backtest_results['avg_win']:.2%}",
                f"{backtest_results['avg_loss']:.2%}",
                f"{backtest_results['profit_factor']:.2f}"
            ]
        })

        # 保存报告
        report.to_csv('backtest_report.csv')
        return report

# config/config.py

import os
from datetime import datetime, timedelta


class Config:
    # 基础配置
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
    MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

    # Tushare配置
    TUSHARE_TOKEN = "你的token"

    # 数据配置
    START_DATE = (datetime.now() - timedelta(days=365 * 5)).strftime('%Y%m%d')  # 5年数据
    END_DATE = datetime.now().strftime('%Y%m%d')

    # 股票池配置
    EXCLUDE_ST = True
    MIN_PRICE = 5.0

    # 交易配置
    TRADE_TYPE = "daily"  # daily/intraday
    COMMISSION_RATE = 0.0003  # 手续费率
    SLIPPAGE_RATE = 0.0001  # 滑点率
    POSITION_LIMIT = 0.8  # 最大仓位限制
    MIN_HOLDINGS = 1  # 最小持仓数
    MAX_HOLDINGS = 10  # 最大持仓数

    # 模型配置
    SEQUENCE_LENGTH = 30
    BATCH_SIZE = 64
    EPOCHS = 100
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    DROPOUT = 0.2

    # 风险控制
    MAX_DRAWDOWN = 0.15
    STOP_LOSS = 0.05

    # 回测配置
    INITIAL_CAPITAL = 1000000
    BENCHMARK = '000300.SH'  # 沪深300作为基准

    # 预警配置
    ALERT_METHODS = ['log', 'email']  # 预警方式
    EMAIL_SETTINGS = {
        'smtp_server': 'smtp.example.com',
        'smtp_port': 587,
        'sender_email': 'your_email@example.com',
        'password': 'your_password'
    }

    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        for directory in [cls.DATA_DIR, cls.LOG_DIR, cls.MODEL_DIR]:
            if not os.path.exists(directory):
                os.makedirs(directory)


# data/data_engine.py

import tushare as ts
import pandas as pd
from datetime import datetime, timedelta
from ..config.config import Config


class DataEngine:
    def __init__(self):
        self.pro = ts.pro_api(Config.TUSHARE_TOKEN)

    def get_stock_list(self):
        """获取股票列表"""
        try:
            stocks = self.pro.stock_basic(
                exchange='',
                list_status='L',
                fields='ts_code,symbol,name,area,industry,list_date'
            )

            if Config.EXCLUDE_ST:
                stocks = stocks[~stocks['name'].str.contains('ST')]

            return stocks
        except Exception as e:
            print(f"获取股票列表失败: {str(e)}")
            return None

    def get_daily_data(self, ts_code, start_date=None, end_date=None):
        """获取日线数据"""
        try:
            if start_date is None:
                start_date = Config.START_DATE
            if end_date is None:
                end_date = Config.END_DATE

            daily_data = self.pro.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields='ts_code,trade_date,open,high,low,close,vol,amount'
            )

            return daily_data
        except Exception as e:
            print(f"获取日线数据失败: {str(e)}")
            return None

    def get_minute_data(self, ts_code, trade_date):
        """获取分钟级数据"""
        try:
            df = self.pro.stk_mins(
                ts_code=ts_code,
                trade_date=trade_date
            )
            return df
        except Exception as e:
            print(f"获取分钟数据失败: {str(e)}")
            return None


# data/data_processor.py

import pandas as pd
import numpy as np


class DataProcessor:
    @staticmethod
    def process_daily_data(df):
        """处理日线数据"""
        if df is None or df.empty:
            return None

        # 按日期排序
        df = df.sort_values('trade_date')

        # 处理缺失值
        df = df.fillna(method='ffill')

        # 添加基础计算列
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        return df

    @staticmethod
    def process_minute_data(df):
        """处理分钟数据"""
        if df is None or df.empty:
            return None

        # 处理分钟数据的特定逻辑
        df = df.sort_values('trade_time')
        df = df.fillna(method='ffill')

        return df


# data/feature_engineer.py

import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import StandardScaler
import logging


class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []

    def create_features(self, df):
        """
        创建特征
        包括：技术指标、波动性指标、趋势指标等
        """
        try:
            features = pd.DataFrame(index=df.index)

            # 基础价格特征
            features = self._add_price_features(df, features)

            # 技术指标
            features = self._add_technical_indicators(df, features)

            # 波动性指标
            features = self._add_volatility_indicators(df, features)

            # 趋势特征
            features = self._add_trend_features(df, features)

            # 成交量特征
            features = self._add_volume_features(df, features)

            # 去除NaN值
            features = features.dropna()

            # 保存特征名称
            self.feature_names = features.columns.tolist()

            return features

        except Exception as e:
            logging.error(f"特征创建失败: {str(e)}")
            return None

    def _add_price_features(self, df, features):
        """价格相关特征"""
        # 价格差异
        features['price_range'] = df['high'] - df['low']
        features['price_range_ratio'] = features['price_range'] / df['close']

        # 价格变化
        features['price_change'] = df['close'].pct_change()
        features['price_change_5'] = df['close'].pct_change(5)
        features['price_change_10'] = df['close'].pct_change(10)
        features['price_change_20'] = df['close'].pct_change(20)

        # 对数收益率
        features['log_return'] = np.log(df['close'] / df['close'].shift(1))

        return features

    def _add_technical_indicators(self, df, features):
        """技术指标"""
        # 移动平均线
        for period in [5, 10, 20, 30, 60]:
            features[f'ma_{period}'] = talib.MA(df['close'], timeperiod=period)
            features[f'ma_diff_{period}'] = df['close'] - features[f'ma_{period}']

        # MACD
        features['macd'], features['macd_signal'], features['macd_hist'] = talib.MACD(
            df['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )

        # RSI
        for period in [6, 12, 24]:
            features[f'rsi_{period}'] = talib.RSI(df['close'], timeperiod=period)

        # KDJ
        features['slowk'], features['slowd'] = talib.STOCH(
            df['high'], df['low'], df['close'],
            fastk_period=9, slowk_period=3, slowk_matype=0,
            slowd_period=3, slowd_matype=0
        )

        # 布林带
        features['bb_upper'], features['bb_middle'], features['bb_lower'] = talib.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )

        return features

    def _add_volatility_indicators(self, df, features):
        """波动性指标"""
        # ATR - Average True Range
        features['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)

        # 历史波动率
        features['volatility_5'] = df['close'].rolling(window=5).std()
        features['volatility_10'] = df['close'].rolling(window=10).std()
        features['volatility_20'] = df['close'].rolling(window=20).std()

        return features

    def _add_trend_features(self, df, features):
        """趋势特征"""
        # ADX - Average Directional Index
        features['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)

        # AROON
        features['aroon_up'], features['aroon_down'] = talib.AROON(df['high'], df['low'], timeperiod=14)

        # CCI - Commodity Channel Index
        features['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)

        return features

    def _add_volume_features(self, df, features):
        """成交量特征"""
        # 成交量变化
        features['volume_change'] = df['volume'].pct_change()

        # 成交量移动平均
        for period in [5, 10, 20]:
            features[f'volume_ma_{period}'] = talib.MA(df['volume'], timeperiod=period)

        # 成交量相对强度
        features['volume_rsi'] = talib.RSI(df['volume'], timeperiod=14)

        # OBV - On Balance Volume
        features['obv'] = talib.OBV(df['close'], df['volume'])

        return features

    def normalize_features(self, features):
        """特征标准化"""
        normalized_features = pd.DataFrame(
            self.scaler.fit_transform(features),
            columns=features.columns,
            index=features.index
        )
        return normalized_features


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


# models/model_engine.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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



# models/model_evaluator.py
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging


class ModelEvaluator:
    def __init__(self):
        self.evaluation_results = {}

    def evaluate_model(self, model, test_data, test_labels):
        """评估模型性能"""
        try:
            # 获取预测结果
            predictions = model.predict(test_data)
            pred_labels = np.argmax(predictions, axis=1)

            # 计算评估指标
            metrics = {
                'accuracy': accuracy_score(test_labels, pred_labels),
                'precision': precision_score(test_labels, pred_labels, average='weighted'),
                'recall': recall_score(test_labels, pred_labels, average='weighted'),
                'f1': f1_score(test_labels, pred_labels, average='weighted')
            }

            # 保存评估结果
            self.evaluation_results = {
                'metrics': metrics,
                'predictions': predictions,
                'true_labels': test_labels,
                'timestamp': datetime.now()
            }

            return metrics

        except Exception as e:
            logging.error(f"模型评估失败: {str(e)}")
            return None

    def plot_learning_curves(self, train_losses, valid_losses):
        """绘制学习曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(valid_losses, label='Validation Loss')
        plt.title('Learning Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('learning_curves.png')
        plt.close()

    def plot_confusion_matrix(self, true_labels, pred_labels):
        """绘制混淆矩阵"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(pd.crosstab(true_labels, pred_labels, normalize='index'),
                    annot=True, fmt='.2%', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('confusion_matrix.png')
        plt.close()

    def analyze_predictions(self):
        """分析预测结果"""
        if not self.evaluation_results:
            return None

        analysis = {
            'prediction_distribution': np.bincount(
                np.argmax(self.evaluation_results['predictions'], axis=1)
            ),
            'confidence_scores': {
                'mean': np.mean(np.max(self.evaluation_results['predictions'], axis=1)),
                'std': np.std(np.max(self.evaluation_results['predictions'], axis=1))
            }
        }

        return analysis

    def generate_report(self):
        """生成评估报告"""
        if not self.evaluation_results:
            return None

        report = {
            'metrics': self.evaluation_results['metrics'],
            'analysis': self.analyze_predictions(),
            'timestamp': self.evaluation_results['timestamp']
        }

        # 保存报告
        report_path = f'model_evaluation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4, default=str)

        return report




# models/parameter_optimizer.py

import itertools
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import ParameterGrid
import optuna
import logging
from datetime import datetime
import json


class ParameterOptimizer:
    def __init__(self, model_engine, data_engine, evaluator):
        self.model_engine = model_engine
        self.data_engine = data_engine
        self.evaluator = evaluator
        self.best_params = None
        self.optimization_history = []

    def grid_search(self, param_grid, train_data, valid_data):
        """
        网格搜索优化

        Parameters:
        param_grid (dict): 参数网格
        train_data: 训练数据
        valid_data: 验证数据
        """
        try:
            best_score = float('-inf')
            best_params = None

            # 创建参数组合
            param_combinations = ParameterGrid(param_grid)
            total_combinations = len(param_combinations)

            logging.info(f"开始网格搜索，共{total_combinations}种参数组合")

            for i, params in enumerate(param_combinations, 1):
                logging.info(f"测试参数组合 {i}/{total_combinations}: {params}")

                # 训练模型
                self.model_engine.train(train_data, valid_data, model_params=params)

                # 评估模型
                metrics = self.evaluator.evaluate_model(
                    self.model_engine.model,
                    valid_data.features,
                    valid_data.labels
                )

                # 记录结果
                result = {
                    'params': params,
                    'metrics': metrics,
                    'timestamp': datetime.now()
                }
                self.optimization_history.append(result)

                # 更新最佳参数
                if metrics['f1'] > best_score:
                    best_score = metrics['f1']
                    best_params = params

            self.best_params = best_params
            self._save_optimization_results()

            return best_params

        except Exception as e:
            logging.error(f"网格搜索失败: {str(e)}")
            return None

    def bayesian_optimization(self, param_space, train_data, valid_data, n_trials=100):
        """
        贝叶斯优化

        Parameters:
        param_space (dict): 参数空间
        train_data: 训练数据
        valid_data: 验证数据
        n_trials (int): 优化迭代次数
        """

        def objective(trial):
            # 从参数空间采样
            params = {
                name: trial._suggest(name, spec)
                for name, spec in param_space.items()
            }

            # 训练模型
            self.model_engine.train(train_data, valid_data, model_params=params)

            # 评估模型
            metrics = self.evaluator.evaluate_model(
                self.model_engine.model,
                valid_data.features,
                valid_data.labels
            )

            # 记录结果
            result = {
                'params': params,
                'metrics': metrics,
                'timestamp': datetime.now()
            }
            self.optimization_history.append(result)

            return metrics['f1']

        try:
            # 创建优化研究
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)

            self.best_params = study.best_params
            self._save_optimization_results()

            return study.best_params

        except Exception as e:
            logging.error(f"贝叶斯优化失败: {str(e)}")
            return None

    def _save_optimization_results(self):
        """保存优化结果"""
        results = {
            'best_params': self.best_params,
            'optimization_history': self.optimization_history
        }

        filename = f'parameter_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4, default=str)

    def plot_optimization_history(self):
        """绘制优化历史"""
        if not self.optimization_history:
            return

        plt.figure(figsize=(12, 6))

        # 提取性能指标
        scores = [result['metrics']['f1'] for result in self.optimization_history]
        trials = range(1, len(scores) + 1)

        # 绘制优化过程
        plt.plot(trials, scores, 'b-', label='F1 Score')
        plt.plot(trials, np.maximum.accumulate(scores), 'r--', label='Best F1 Score')

        plt.xlabel('Trial')
        plt.ylabel('F1 Score')
        plt.title('Parameter Optimization History')
        plt.legend()
        plt.grid(True)

        plt.savefig('optimization_history.png')
        plt.close()


# monitor/monitor_engine.py

import pandas as pd
from datetime import datetime
import logging
import json
from ..config.config import Config


class MonitorEngine:
    def __init__(self):
        self.performance_metrics = []
        self.risk_metrics = []
        self.alerts = []

    def update_performance(self, metrics):
        """
        更新性能指标

        Parameters:
        metrics (dict): 性能指标
        """
        metrics['timestamp'] = datetime.now()
        self.performance_metrics.append(metrics)

        # 检查性能预警
        self._check_performance_alerts(metrics)

    def update_risk_metrics(self, metrics):
        """
        更新风险指标

        Parameters:
        metrics (dict): 风险指标
        """
        metrics['timestamp'] = datetime.now()
        self.risk_metrics.append(metrics)

        # 检查风险预警
        self._check_risk_alerts(metrics)

    def add_alert(self, alert_type, message, severity='INFO'):
        """添加预警信息"""
        alert = {
            'type': alert_type,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now()
        }

        self.alerts.append(alert)

        # 记录日志
        if severity == 'ERROR':
            logging.error(message)
        elif severity == 'WARNING':
            logging.warning(message)
        else:
            logging.info(message)

    def _check_performance_alerts(self, metrics):
        """检查性能预警"""
        # 检查回撤
        if metrics.get('drawdown', 0) > Config.MAX_DRAWDOWN:
            self.add_alert(
                'DRAWDOWN',
                f"Maximum drawdown exceeded: {metrics['drawdown']:.2%}",
                'WARNING'
            )

        # 检查收益率
        if metrics.get('daily_return', 0) < -0.05:  # 单日跌幅超过5%
            self.add_alert(
                'RETURN',
                f"Significant daily loss: {metrics['daily_return']:.2%}",
                'WARNING'
            )

    def _check_risk_alerts(self, metrics):
        """检查风险预警"""
        # 检查持仓集中度
        if metrics.get('concentration_risk', 0) > 0.5:  # 单个持仓超过50%
            self.add_alert(
                'CONCENTRATION',
                f"High position concentration: {metrics['concentration_risk']:.2%}",
                'WARNING'
            )

    def get_summary(self):
        """获取监控摘要"""
        return {
            'latest_performance': self.performance_metrics[-1] if self.performance_metrics else None,
            'latest_risk': self.risk_metrics[-1] if self.risk_metrics else None,
            'recent_alerts': self.alerts[-10:]  # 最近10条预警
        }

    def export_report(self, filepath):
        """导出监控报告"""
        report = {
            'performance_metrics': self.performance_metrics,
            'risk_metrics': self.risk_metrics,
            'alerts': self.alerts
        }

        with open(filepath, 'w') as f:
            json.dump(report, f, default=str, indent=4)



# risk/risk_manager.py

import pandas as pd
import numpy as np
from ..config.config import Config
import logging


class RiskManager:
    def __init__(self):
        self.positions = {}  # 当前持仓
        self.position_values = {}  # 持仓市值
        self.total_value = 0  # 总资产
        self.stop_loss_prices = {}  # 止损价格

    def update_portfolio(self, positions, current_prices):
        """
        更新投资组合状态

        Parameters:
        positions (dict): 当前持仓股票及数量
        current_prices (dict): 当前股票价格
        """
        self.positions = positions.copy()
        self.position_values = {}
        self.total_value = 0

        # 更新持仓市值
        for symbol, shares in positions.items():
            if symbol in current_prices:
                value = shares * current_prices[symbol]
                self.position_values[symbol] = value
                self.total_value += value

    def check_position_limit(self, symbol, shares, price):
        """
        检查是否超出持仓限制

        Returns:
        bool: 是否允许交易
        """
        # 计算交易后的持仓市值
        potential_value = shares * price
        current_total = sum(self.position_values.values())

        # 检查单个持仓限制
        if potential_value / (current_total + potential_value) > Config.POSITION_LIMIT:
            logging.warning(f"Position limit exceeded for {symbol}")
            return False

        return True

    def set_stop_loss(self, symbol, entry_price):
        """设置止损价格"""
        self.stop_loss_prices[symbol] = entry_price * (1 - Config.STOP_LOSS)

    def check_stop_loss(self, symbol, current_price):
        """
        检查是否触发止损

        Returns:
        bool: 是否需要止损
        """
        if symbol in self.stop_loss_prices:
            if current_price <= self.stop_loss_prices[symbol]:
                logging.warning(f"Stop loss triggered for {symbol}")
                return True
        return False

    def check_drawdown(self, portfolio_values):
        """
        检查是否超过最大回撤限制

        Returns:
        bool: 是否需要降低仓位
        """
        if len(portfolio_values) < 2:
            return False

        # 计算当前回撤
        peak = max(portfolio_values)
        current_value = portfolio_values[-1]
        drawdown = (peak - current_value) / peak

        if drawdown > Config.MAX_DRAWDOWN:
            logging.warning(f"Maximum drawdown exceeded: {drawdown:.2%}")
            return True

        return False

    def get_position_adjustment(self):
        """
        获取仓位调整建议

        Returns:
        dict: 需要调整的仓位
        """
        adjustments = {}

        # 检查是否需要调整仓位
        for symbol, value in self.position_values.items():
            position_ratio = value / self.total_value
            if position_ratio > Config.POSITION_LIMIT:
                # 计算需要减少的份额
                excess_value = value - (self.total_value * Config.POSITION_LIMIT)
                shares_to_reduce = int(excess_value / (value / self.positions[symbol]))
                adjustments[symbol] = -shares_to_reduce

        return adjustments

    def get_risk_metrics(self):
        """
        获取风险指标

        Returns:
        dict: 风险指标
        """
        metrics = {
            'total_exposure': sum(self.position_values.values()),
            'largest_position': max(self.position_values.values()) if self.position_values else 0,
            'position_count': len(self.positions),
            'concentration_risk': max(
                [v / self.total_value for v in self.position_values.values()]) if self.position_values else 0
        }

        return metrics



# trading/trade_engine.py

import pandas as pd
import numpy as np
from datetime import datetime
from ..config.config import Config
from ..risk.risk_manager import RiskManager
import logging


class TradeEngine:
    def __init__(self, data_engine, model_engine, risk_manager):
        self.data_engine = data_engine
        self.model_engine = model_engine
        self.risk_manager = risk_manager
        self.positions = {}
        self.pending_orders = []
        self.executed_orders = []

    def generate_signals(self, current_data):
        """
        生成交易信号

        Parameters:
        current_data (DataFrame): 当前市场数据

        Returns:
        dict: 交易信号
        """
        try:
            # 使用模型预测
            predictions = self.model_engine.predict(current_data)

            # 生成交易信号
            signals = {}
            for symbol, pred in predictions.items():
                if pred > 0.7:  # 买入阈值
                    signals[symbol] = 1
                elif pred < 0.3:  # 卖出阈值
                    signals[symbol] = -1
                else:
                    signals[symbol] = 0

            return signals

        except Exception as e:
            logging.error(f"Error generating signals: {str(e)}")
            return {}

    def execute_trades(self, signals, current_prices):
        """
        执行交易

        Parameters:
        signals (dict): 交易信号
        current_prices (dict): 当前价格
        """
        try:
            for symbol, signal in signals.items():
                # 检查风险控制
                if not self.risk_manager.check_position_limit(symbol, 100, current_prices[symbol]):
                    continue

                # 检查止损
                if symbol in self.positions and self.risk_manager.check_stop_loss(symbol, current_prices[symbol]):
                    self._place_order(symbol, -self.positions[symbol], current_prices[symbol], 'SELL')
                    continue

                if signal > 0:  # 买入信号
                    if symbol not in self.positions:
                        shares = self._calculate_position_size(current_prices[symbol])
                        self._place_order(symbol, shares, current_prices[symbol], 'BUY')

                elif signal < 0:  # 卖出信号
                    if symbol in self.positions:
                        self._place_order(symbol, -self.positions[symbol], current_prices[symbol], 'SELL')

        except Exception as e:
            logging.error(f"Error executing trades: {str(e)}")

    def _place_order(self, symbol, shares, price, order_type):
        """下单"""
        order = {
            'symbol': symbol,
            'shares': shares,
            'price': price,
            'type': order_type,
            'timestamp': datetime.now(),
            'status': 'PENDING'
        }

        self.pending_orders.append(order)
        logging.info(f"Order placed: {order}")

    def _calculate_position_size(self, price):
        """计算仓位大小"""
        # 这里可以实现更复杂的仓位计算逻辑
        return 100  # 暂时固定为100股

    def update_positions(self):
        """更新持仓状态"""
        for order in self.pending_orders:
            if order['status'] == 'PENDING':
                if order['type'] == 'BUY':
                    self.positions[order['symbol']] = self.positions.get(order['symbol'], 0) + order['shares']
                else:
                    self.positions[order['symbol']] = self.positions.get(order['symbol'], 0) - order['shares']

                order['status'] = 'EXECUTED'
                self.executed_orders.append(order)

        self.pending_orders = [order for order in self.pending_orders if order['status'] == 'PENDING']

    def get_positions_summary(self):
        """获取持仓摘要"""
        return {
            'positions': self.positions.copy(),
            'pending_orders': len(self.pending_orders),
            'executed_orders': len(self.executed_orders)
        }


# utils/logger.py

import logging
import os
from datetime import datetime


def setup_logger(log_dir='logs'):
    """
    设置日志配置

    Parameters:
    log_dir (str): 日志文件目录
    """
    # 创建日志目录
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 生成日志文件名
    log_file = os.path.join(log_dir, f'quant_trading_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # 设置第三方库的日志级别
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)




import logging

from lh_code_adv.backtesting.backtesting_engine import BacktestEngine
from lh_code_adv.config.config import Config
from lh_code_adv.data.data_engine import DataEngine
from lh_code_adv.data.data_processor import DataProcessor
from lh_code_adv.data.feature_engineer import FeatureEngineer
from lh_code_adv.models.model_engine import ModelEngine


class QuantTradingSystem:
    def __init__(self):
        self.data_engine = DataEngine()
        self.data_processor = DataProcessor()
        self.feature_engineer = FeatureEngineer()
        self.model_engine = ModelEngine()
        self.backtest_engine = BacktestEngine(Config.INITIAL_CAPITAL)

    def run_training(self):
        """
        模型训练流程:
        1. 获取历史数据
        2. 特征工程
        3. 模型训练
        4. 模型评估
        """
        try:
            logging.info("开始模型训练...")

            # 1. 获取训练数据
            train_data = self._prepare_training_data()

            # 2. 特征工程
            features = self._engineer_features(train_data)

            # 3. 划分训练集和验证集
            train_dataset, valid_dataset = self._split_dataset(features)

            # 4. 模型训练
            model_performance = self.model_engine.train(train_dataset, valid_dataset)

            # 5. 保存模型和训练结果
            self._save_training_results(model_performance)

            return model_performance

        except Exception as e:
            logging.error(f"模型训练失败: {str(e)}")
            raise

    def _prepare_training_data(self):
        """准备训练数据"""
        stocks = self.data_engine.get_stock_list()
        training_data = {}

        for ts_code in stocks['ts_code']:
            data = self.data_engine.get_daily_data(ts_code)
            if data is not None and not data.empty:
                training_data[ts_code] = data

        return training_data

    def _engineer_features(self, data):
        """特征工程"""
        features = {}
        for ts_code, stock_data in data.items():
            # 数据预处理
            processed_data = self.data_processor.process_daily_data(stock_data)
            # 创建技术指标
            featured_data = self.feature_engineer.create_technical_features(processed_data)
            if featured_data is not None:
                features[ts_code] = featured_data
        return features

    def run_backtest(self, model_path=None):
        """
        回测流程:
        1. 加载训练好的模型
        2. 获取回测数据
        3. 生成交易信号
        4. 执行回测
        5. 分析回测结果
        """
        try:
            logging.info("开始回测...")

            # 1. 加载模型（如果指定了模型路径）
            if model_path:
                self.model_engine.load_model(model_path)

            # 2. 获取回测数据
            backtest_data = self._prepare_backtest_data()

            # 3. 生成交易信号
            signals = self._generate_trading_signals(backtest_data)

            # 4. 执行回测
            results = self.backtest_engine.run_backtest(signals, backtest_data)

            # 5. 分析和保存回测结果
            self._analyze_backtest_results(results)

            return results

        except Exception as e:
            logging.error(f"回测失败: {str(e)}")
            raise

    def model_evaluation(self):
        """模型评估"""
        results = {}

        # 测试不同的模型参数
        param_combinations = [
            {'hidden_dim': 64, 'num_layers': 2},
            {'hidden_dim': 128, 'num_layers': 2},
            {'hidden_dim': 128, 'num_layers': 3},
            # 更多参数组合...
        ]

        for params in param_combinations:
            # 训练模型
            model_performance = self.run_training(model_params=params)
            # 运行回测
            backtest_results = self.run_backtest()

            results[str(params)] = {
                'model_performance': model_performance,
                'backtest_results': backtest_results
            }

        return results