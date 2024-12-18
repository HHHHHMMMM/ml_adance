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