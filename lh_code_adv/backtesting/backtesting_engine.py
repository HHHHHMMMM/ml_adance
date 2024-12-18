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