from typing import Dict

import numpy as np


class Backtester:
    def __init__(self, initial_capital=1000000, params=None):
        """
        初始化回测器

        Parameters:
        -----------
        initial_capital : float
            初始资金
        params : dict
            回测参数
        """
        # 资金相关
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital  # 记录最高资金量
        self.max_drawdown = 0  # 最大回撤

        # 交易记录
        self.trades = []
        self.positions: Dict[str, Dict] = {}
        self.daily_records = []  # 每日状态记录

        # 设置默认参数
        default_params = {
            'trade_cost': 0.0003,  # 交易成本
            'slippage': 0.002,  # 滑点
            'max_hold_days': 10,  # 最大持仓天数
            'max_position_size': 0.2,  # 最大单个持仓比例
            'max_loss_per_trade': 0.02,  # 单笔最大亏损
            'profit_target': 0.03,  # 目标收益
            'max_positions': 5,  # 最大同时持仓数
            'risk_free_rate': 0.03,  # 无风险利率
            'stop_loss_pct': 0.02,  # 止损比例
        }

        # 更新参数
        self.params = default_params
        if params:
            self.params.update(params)

        # 设置参数为实例属性
        for key, value in self.params.items():
            setattr(self, key, value)

        # 性能指标
        self.performance_metrics = {}

    def reset(self):
        """重置回测状态"""
        self.current_capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.max_drawdown = 0
        self.trades = []
        self.positions = {}
        self.daily_records = []
        self.performance_metrics = {}

    def backtest(self, predictions, price_data, params=None):
        """
        执行回测

        Parameters:
        -----------
        predictions : array-like
            交易信号预测
        price_data : pd.DataFrame
            价格数据
        params : dict, optional
            回测参数
        """
        # 重置回测状态
        self.reset()

        # 更新参数
        if params:
            self.params.update(params)
            for key, value in params.items():
                setattr(self, key, value)

        returns = []
        daily_positions = []

        for i in range(len(predictions)):
            current_price = price_data.iloc[i]

            # 记录每日状态
            daily_state = {
                'date': current_price.name,
                'capital': self.current_capital,
                'positions': len(self.positions),
                'price': current_price['close']
            }
            self.daily_records.append(daily_state)

            # 更新最大回撤
            self.peak_capital = max(self.peak_capital, self.current_capital)
            current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
            self.max_drawdown = max(self.max_drawdown, current_drawdown)

            if i == 0:
                returns.append(0)
                daily_positions.append(0)
                continue

            prev_price = price_data.iloc[i - 1]

            # 更新现有持仓
            self._update_positions(current_price)

            # 检查是否可以开新仓
            can_open_position = (
                    len(self.positions) < self.max_positions and
                    self.current_capital > self.initial_capital * 0.1  # 确保至少有10%资金可用
            )

            # 处理交易信号
            if predictions[i] == 1 and can_open_position:  # 买入信号
                self._handle_buy_signal(current_price)
            elif predictions[i] == 2:  # 卖出信号
                self._handle_sell_signal(current_price)

            # 计算当日收益
            daily_return = self._calculate_daily_return(current_price, prev_price)
            returns.append(daily_return)

            # 记录持仓情况
            daily_positions.append(len(self.positions))

        # 计算性能指标
        self._calculate_performance_metrics(returns)

        # 生成回测报告
        backtest_results = {
            'returns': np.array(returns),
            'trades': self.trades,
            'daily_positions': daily_positions,
            'daily_records': self.daily_records,
            'final_capital': self.current_capital,
            'return_rate': (self.current_capital - self.initial_capital) / self.initial_capital,
            'max_drawdown': self.max_drawdown,
            'trade_statistics': self._calculate_trade_statistics(),
            'performance_metrics': self.performance_metrics
        }

        return backtest_results


    def _update_positions(self, current_price):
        """更新持仓状态"""
        positions_to_close = []
        for symbol, position in self.positions.items():
            # 更新持仓天数
            position['hold_days'] += 1

            # 计算当前收益
            current_return = (current_price['close'] / position['entry_price']) - 1

            # 添加止损检查
            if current_return <= -self.stop_loss_pct:
                positions_to_close.append((symbol, 'stop_loss'))
                continue

            # 检查其他平仓条件
            if (position['hold_days'] >= self.max_hold_days or  # 超过最大持仓时间
                    current_return <= -self.max_loss_per_trade or  # 达到止损
                    current_return >= self.profit_target):  # 达到止盈
                positions_to_close.append((symbol, 'auto'))

        # 平仓
        for symbol, reason in positions_to_close:
            self._close_position(symbol, current_price, reason)


    def _handle_buy_signal(self, current_price):
        """处理买入信号"""
        # 检查是否有足够资金和仓位空间
        if (len(self.positions) * self.max_position_size < 1.0 and
                self.current_capital > 0):
            # 计算可用仓位
            position_size = min(
                self.max_position_size,
                self.current_capital * 0.2  # 最多使用20%可用资金
            )

            # 创建新仓位
            position = {
                'entry_price': current_price['close'],
                'size': position_size,
                'hold_days': 0,
                'entry_time': current_price.name,  # 假设使用时间索引
                'cost': current_price['close'] * position_size * (1 + self.trade_cost + self.slippage)
            }

            # 更新资金和持仓
            self.current_capital -= position['cost']
            self.positions[f"pos_{len(self.trades)}"] = position

            # 记录交易
            self.trades.append({
                'type': 'buy',
                'time': current_price.name,
                'price': current_price['close'],
                'size': position_size,
                'cost': position['cost']
            })

    def _handle_sell_signal(self, current_price):
        """处理卖出信号"""
        for symbol in list(self.positions.keys()):
            self._close_position(symbol, current_price, 'signal')

    def _close_position(self, symbol, current_price, reason):
        """平仓操作"""
        position = self.positions[symbol]

        # 计算收益
        revenue = (current_price['close'] * position['size'] *
                   (1 - self.trade_cost - self.slippage))
        profit = revenue - position['cost']

        # 更新资金
        self.current_capital += revenue

        # 记录交易
        self.trades.append({
            'type': 'sell',
            'time': current_price.name,
            'price': current_price['close'],
            'size': position['size'],
            'revenue': revenue,
            'profit': profit,
            'hold_days': position['hold_days'],
            'reason': reason
        })

        # 移除持仓
        del self.positions[symbol]

    def _calculate_daily_return(self, current_price, prev_price):
        """计算每日收益率"""
        if not self.positions:
            return 0

        total_return = 0
        for position in self.positions.values():
            daily_return = (current_price['close'] / prev_price['close'] - 1) * position['size']
            total_return += daily_return

        return total_return

    def _calculate_trade_statistics(self):
        """计算交易统计指标"""
        if not self.trades:
            return {}

        profits = [t['profit'] for t in self.trades if 'profit' in t]
        hold_days = [t['hold_days'] for t in self.trades if 'hold_days' in t]

        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p < 0]

        return {
            'total_trades': len(profits),
            'win_rate': len(winning_trades) / len(profits) if profits else 0,
            'avg_profit': np.mean(profits) if profits else 0,
            'avg_winning_trade': np.mean(winning_trades) if winning_trades else 0,
            'avg_losing_trade': np.mean(losing_trades) if losing_trades else 0,
            'max_profit': max(profits) if profits else 0,
            'max_loss': min(profits) if profits else 0,
            'avg_hold_days': np.mean(hold_days) if hold_days else 0,
            'profit_factor': abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else float('inf'),
            'win_loss_ratio': abs(
                np.mean(winning_trades) / np.mean(losing_trades)) if losing_trades and winning_trades else 0
        }
    def _calculate_performance_metrics(self, returns):
        """计算性能指标"""
        returns_array = np.array(returns)

        # 计算年化收益率（假设252个交易日）
        total_days = len(returns)
        annual_return = (1 + sum(returns)) ** (252 / total_days) - 1

        # 计算夏普比率
        daily_rf = (1 + self.risk_free_rate) ** (1 / 252) - 1
        excess_returns = returns_array - daily_rf
        sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(returns_array) if np.std(
            returns_array) != 0 else 0

        self.performance_metrics = {
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'volatility': np.std(returns_array) * np.sqrt(252),
            'total_trades': len(self.trades),
            'win_rate': sum(1 for t in self.trades if t.get('profit', 0) > 0) / len(self.trades) if self.trades else 0
        }