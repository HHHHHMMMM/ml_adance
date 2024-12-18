import itertools

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


class StrategyEvaluator:
    def __init__(self):
        self.performance_metrics = {}
        self.risk_metrics = {}

    def evaluate_strategy(self, returns, benchmark_returns=None):
        """评估策略表现"""
        if len(returns) == 0:
            return {
                'performance': {
                    'cumulative_return': 0,
                    'annual_return': 0,
                    'win_rate': 0
                },
                'risk': {
                    'volatility': 0,
                    'max_drawdown': 0,
                    'sharpe_ratio': 0
                }
            }

        self.performance_metrics = self._calculate_performance_metrics(returns, benchmark_returns)
        self.risk_metrics = self._calculate_risk_metrics(returns)

        return {
            'performance': self.performance_metrics,
            'risk': self.risk_metrics
        }

    def _calculate_performance_metrics(self, returns, benchmark_returns):
        """计算性能指标"""
        metrics = {}

        # 确保returns是numpy数组
        returns = np.array(returns)

        # 处理基准收益率
        if benchmark_returns is not None:
            # 确保基准收益率是numpy数组
            benchmark_returns = np.array(benchmark_returns)

            # 确保两个序列长度一致
            min_length = min(len(returns), len(benchmark_returns))
            returns = returns[:min_length]
            benchmark_returns = benchmark_returns[:min_length]

            # 计算超额收益
            excess_returns = returns - benchmark_returns
            metrics['information_ratio'] = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)

        # 累积收益
        metrics['cumulative_return'] = (1 + returns).prod() - 1

        # 年化收益
        days = len(returns)
        metrics['annual_return'] = (1 + metrics['cumulative_return']) ** (252 / days) - 1

        # 夏普比率（假设无风险利率为3%）
        risk_free_rate = 0.03  # 年化3%
        excess_returns = returns - risk_free_rate / 252  # 转换为日收益率
        metrics['sharpe_ratio'] = np.sqrt(252) * np.mean(excess_returns) / np.std(returns)

        # 最大回撤
        cum_returns = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = (cum_returns - running_max) / running_max
        metrics['max_drawdown'] = abs(drawdowns.min())

        # 胜率
        metrics['win_rate'] = len(returns[returns > 0]) / len(returns)

        # 盈亏比
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            metrics['profit_loss_ratio'] = abs(np.mean(positive_returns) / np.mean(negative_returns))
        else:
            metrics['profit_loss_ratio'] = float('inf')

        return metrics
    def _calculate_risk_metrics(self, returns):
        """计算风险指标"""
        metrics = {}

        # 波动率
        metrics['volatility'] = np.std(returns) * np.sqrt(252)

        # 最大回撤
        cum_returns = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = (cum_returns - running_max) / running_max
        metrics['max_drawdown'] = np.min(drawdowns)

        # 夏普比率
        risk_free_rate = 0.03  # 假设无风险利率为3%
        excess_returns = returns - risk_free_rate / 252
        metrics['sharpe_ratio'] = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

        # 索提诺比率
        downside_returns = returns[returns < 0]
        metrics['sortino_ratio'] = np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)

        return metrics

    def plot_performance(self, returns, benchmark_returns=None):
        """绘制策略表现图表"""
        plt.figure(figsize=(15, 10))

        # 累积收益图
        plt.subplot(2, 2, 1)
        cum_returns = (1 + returns).cumprod()
        plt.plot(cum_returns.index, cum_returns.values, label='Strategy')
        if benchmark_returns is not None:
            cum_benchmark = (1 + benchmark_returns).cumprod()
            plt.plot(cum_benchmark.index, cum_benchmark.values, label='Benchmark')
        plt.title('Cumulative Returns')
        plt.legend()

        # 回撤图
        plt.subplot(2, 2, 2)
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = (cum_returns - running_max) / running_max
        plt.plot(drawdowns.index, drawdowns.values)
        plt.title('Drawdowns')

        # 收益分布图
        plt.subplot(2, 2, 3)
        sns.histplot(returns, kde=True)
        plt.title('Returns Distribution')

        # 月度收益热力图
        plt.subplot(2, 2, 4)
        monthly_returns = returns.resample('M').agg(lambda x: (1 + x).prod() - 1)
        monthly_returns = monthly_returns.groupby(
            [monthly_returns.index.year, monthly_returns.index.month]).first().unstack()
        sns.heatmap(monthly_returns, annot=True, fmt='.2%', cmap='RdYlGn')
        plt.title('Monthly Returns')

        plt.tight_layout()
        plt.show()

    def analyze_trades(self, trades_df):
        """分析交易记录"""
        analysis = {}

        # 交易统计
        analysis['total_trades'] = len(trades_df)
        analysis['winning_trades'] = len(trades_df[trades_df['profit'] > 0])
        analysis['losing_trades'] = len(trades_df[trades_df['profit'] < 0])

        if analysis['total_trades'] > 0:
            analysis['win_rate'] = analysis['winning_trades'] / analysis['total_trades']

            # 盈亏统计
            winning_trades = trades_df[trades_df['profit'] > 0]
            losing_trades = trades_df[trades_df['profit'] < 0]

            analysis['avg_profit'] = winning_trades['profit'].mean() if len(winning_trades) > 0 else 0
            analysis['avg_loss'] = losing_trades['profit'].mean() if len(losing_trades) > 0 else 0
            analysis['profit_factor'] = (
                abs(winning_trades['profit'].sum() / losing_trades['profit'].sum())
                if len(losing_trades) > 0 and losing_trades['profit'].sum() != 0
                else float('inf')
            )

            # 持仓时间分析
            trades_df['holding_period'] = (
                    pd.to_datetime(trades_df['exit_time']) -
                    pd.to_datetime(trades_df['entry_time'])
            )
            analysis['avg_holding_period'] = trades_df['holding_period'].mean()

        return analysis

    def calculate_factor_returns(self, factor_data, returns_data):
        """计算因子收益"""
        # 构建分位数组合
        quantiles = 5
        factor_quantiles = pd.qcut(factor_data, quantiles, labels=False)

        # 计算每个分位数组合的收益
        quantile_returns = {}
        for i in range(quantiles):
            mask = factor_quantiles == i
            quantile_returns[i] = returns_data[mask].mean()

        # 计算多空组合收益
        long_short_returns = quantile_returns[quantiles - 1] - quantile_returns[0]

        return {
            'quantile_returns': quantile_returns,
            'long_short_returns': long_short_returns
        }

    def calculate_position_metrics(self, positions_df):
        """计算持仓指标"""
        metrics = {}

        # 持仓集中度
        metrics['concentration'] = (positions_df['position_value'] / positions_df['position_value'].sum()).max()

        # 行业分布
        metrics['industry_weights'] = positions_df.groupby('industry')['position_value'].sum() / positions_df[
            'position_value'].sum()

        # 持仓市值分布
        metrics['market_cap_distribution'] = positions_df.groupby('market_cap_category')['position_value'].sum() / \
                                             positions_df['position_value'].sum()

        return metrics

    def run_optimization_tests(self, strategy_params, historical_data):
        """运行策略优化测试"""
        results = []

        # 生成参数组合
        param_combinations = self._generate_param_combinations(strategy_params)

        for params in param_combinations:
            # 使用当前参数运行回测
            backtest_result = self._run_backtest_with_params(params, historical_data)

            # 计算性能指标
            metrics = self.evaluate_strategy(backtest_result['returns'])

            results.append({
                'params': params,
                'metrics': metrics
            })

        # 找出最优参数组合
        best_result = max(results, key=lambda x: x['metrics']['performance']['sharpe_ratio'])

        return {
            'all_results': results,
            'best_params': best_result['params'],
            'best_metrics': best_result['metrics']
        }

    def _generate_param_combinations(self, strategy_params):
        """生成参数组合"""
        param_combinations = []

        # 例如: strategy_params = {
        #     'lookback_period': [10, 20, 30],
        #     'holding_period': [5, 10, 15],
        #     'stop_loss': [0.02, 0.03, 0.04]
        # }

        keys = list(strategy_params.keys())
        values = list(strategy_params.values())

        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            param_combinations.append(param_dict)

        return param_combinations

    def _run_backtest_with_params(self, params, historical_data):
        """使用给定参数运行回测"""
        # 实现回测逻辑
        pass
