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