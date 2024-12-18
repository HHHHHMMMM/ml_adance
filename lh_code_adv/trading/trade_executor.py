import pandas as pd

from lh_code_adv.config.config import Config
from lh_code_adv.risk.risk_manager import RiskManager
from lh_code_adv.utils.logger import setup_logger

class TradeExecutor:

    def __init__(self):
        self.positions = {}  # 当前持仓
        self.orders = []  # 订单队列
        self.risk_manager = RiskManager()

    def execute_trades(self, signals, current_prices):
        """执行交易信号"""
        execution_results = {}
        logger = setup_logger('quant_trading')
        for symbol, signal in signals.items():
            try:
                if signal == 0:  # 不交易
                    continue

                # 计算交易数量
                trade_size = self._calculate_trade_size(symbol, signal, current_prices[symbol])

                # 检查风险限制
                if not self.risk_manager.check_position_limit(symbol, trade_size, current_prices[symbol]):
                    logger.warning(f"{symbol} 交易数量超出风险限制，跳过交易")
                    continue

                # 执行交易
                if signal > 0:  # 买入
                    result = self._execute_buy(symbol, trade_size, current_prices[symbol])
                else:  # 卖出
                    result = self._execute_sell(symbol, trade_size, current_prices[symbol])

                execution_results[symbol] = result

            except Exception as e:
                logger.error(f"{symbol} 交易执行失败: {str(e)}")
                execution_results[symbol] = {'status': 'failed', 'error': str(e)}

        return execution_results

    def _calculate_trade_size(self, symbol, signal, current_price):
        """计算交易数量"""
        # 获取账户可用资金
        available_cash = self._get_available_cash()

        # 计算基础交易数量
        if signal > 0:  # 买入
            # 使用可用资金的一定比例
            position_value = available_cash * Config.POSITION_SIZE
            shares = int(position_value / current_price / 100) * 100  # 向下取整到100股
        else:  # 卖出
            # 获取当前持仓
            shares = self.positions.get(symbol, 0)

        return shares

    def _execute_buy(self, symbol, shares, price):
        logger = setup_logger('quant_trading')
        """执行买入订单"""
        try:
            # 检查资金是否足够
            required_capital = shares * price * (1 + Config.COMMISSION_RATE)
            if required_capital > self._get_available_cash():
                return {'status': 'failed', 'error': '资金不足'}

            # 创建订单
            order = {
                'symbol': symbol,
                'type': 'buy',
                'shares': shares,
                'price': price,
                'timestamp': pd.Timestamp.now()
            }

            # 添加到订单队列
            self.orders.append(order)

            # 更新持仓
            self.positions[symbol] = self.positions.get(symbol, 0) + shares

            # 设置止损
            self.risk_manager.set_stop_loss(symbol, price)

            return {'status': 'success', 'order': order}

        except Exception as e:
            logger.error(f"买入执行失败: {str(e)}")
            return {'status': 'failed', 'error': str(e)}

    def _execute_sell(self, symbol, shares, price):
        logger = setup_logger('quant_trading')
        """执行卖出订单"""
        try:
            # 检查持仓是否足够
            current_position = self.positions.get(symbol, 0)
            if current_position < shares:
                return {'status': 'failed', 'error': '持仓不足'}

            # 创建订单
            order = {
                'symbol': symbol,
                'type': 'sell',
                'shares': shares,
                'price': price,
                'timestamp': pd.Timestamp.now()
            }

            # 添加到订单队列
            self.orders.append(order)

            # 更新持仓
            self.positions[symbol] = current_position - shares

            # 如果全部卖出，移除止损价格
            if self.positions[symbol] == 0:
                self.risk_manager.stop_loss_prices.pop(symbol, None)

            return {'status': 'success', 'order': order}

        except Exception as e:
            logger.error(f"卖出执行失败: {str(e)}")
            return {'status': 'failed', 'error': str(e)}

    def _get_available_cash(self):
        """获取可用资金"""
        # 这里需要连接到实际的资金账户系统
        # 临时使用配置中的初始资金
        return Config.INITIAL_CAPITAL