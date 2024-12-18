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