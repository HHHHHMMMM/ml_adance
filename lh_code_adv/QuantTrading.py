import logging
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from lh_code_adv.backtesting.backtesting_engine import BacktestEngine
from lh_code_adv.config.config import Config
from lh_code_adv.data.data_engine import DataEngine
from lh_code_adv.data.data_manager import DataManager
from lh_code_adv.data.data_processor import DataProcessor
from lh_code_adv.data.feature_engineer import FeatureEngineer
from lh_code_adv.data.feature_selector import FeatureSelector
from lh_code_adv.models.model_engine import ModelEngine
from lh_code_adv.models.model_ensemble import ModelEnsemble
from lh_code_adv.monitor.TradingMonitor import TradingMonitor
from lh_code_adv.risk.risk_manager import RiskManager
from lh_code_adv.strategies.strategy_evaluator import StrategyEvaluator
from lh_code_adv.trading.trade_executor import TradeExecutor
from lh_code_adv.utils.logger import log_method, ProgressLogger


class QuantTradingSystem:
    @log_method
    def __init__(self):
        self.logger = logging.getLogger('quant_trading')
        self.logger.info("初始化量化交易系统...")
        self.data_engine = DataEngine()
        self.data_processor = DataProcessor()
        self.feature_engineer = FeatureEngineer()
        self.model_engine = ModelEngine()
        self.backtest_engine = BacktestEngine(Config.INITIAL_CAPITAL)
        self.trade_executor = TradeExecutor()
        self.risk_manager = RiskManager()
        self.trading_monitor = TradingMonitor()
        self._last_trade_time = {}  # 记录每个股票的最后交易时间

    @log_method
    def run_training(self):
        """模型训练流程"""
        self.logger.info("开始模型训练流程...")

        # 1. 获取训练数据
        self.logger.info("Step 1: 获取训练数据")
        train_data = self._prepare_training_data()
        self.logger.info(f"成功获取 {len(train_data)} 只股票的训练数据")

        # 2. 特征工程
        self.logger.info("Step 2: 开始特征工程")
        features = self._engineer_features(train_data)
        self.logger.info(f"特征工程完成，生成 {len(features)} 只股票的特征")

        # 3. 划分训练集和验证集
        self.logger.info("Step 3: 划分数据集")
        train_dataset, valid_dataset = self._split_dataset(features)
        self.logger.info(f"数据集划分完成，训练集大小: {len(train_dataset)}, 验证集大小: {len(valid_dataset)}")

        # 4. 模型训练
        self.logger.info("Step 4: 开始模型训练")
        model_performance = self.model_engine.train(train_dataset, valid_dataset)

        # 5. 保存结果
        self.logger.info("Step 5: 保存训练结果")
        self._save_training_results(model_performance)

        return model_performance

    @log_method
    def _prepare_training_data(self):
        """准备训练数据"""
        stocks = self.data_engine.get_stock_list()
        if stocks is None or len(stocks) == 0:
            self.logger.error("获取股票列表失败")
            raise ValueError("无法获取股票列表")

        self.logger.info(f"获取到 {len(stocks)} 只股票")
        training_data = {}

        progress = ProgressLogger(
            total=len(stocks),
            desc="训练数据获取进度"
        )

        for ts_code in stocks['ts_code']:
            try:
                self.logger.debug(f"正在获取 {ts_code} 的数据...")
                data = self.data_engine.get_daily_data(ts_code)
                if data is not None and not data.empty:
                    training_data[ts_code] = data
                    self.logger.debug(f"{ts_code}: 成功获取 {len(data)} 条数据")
                else:
                    self.logger.warning(f"{ts_code}: 未获取到数据")
            except Exception as e:
                self.logger.error(f"{ts_code} 数据获取失败: {str(e)}")

            progress.update()

        progress.close()
        return training_data

    @log_method
    def _engineer_features(self, data):
        """特征工程"""
        self.logger.info("开始特征工程...")
        features = {}

        progress = ProgressLogger(
            total=len(data),
            desc="特征工程进度"
        )

        for ts_code, stock_data in data.items():
            try:
                # 数据预处理
                self.logger.debug(f"{ts_code}: 开始数据预处理...")
                processed_data = self.data_processor.process_daily_data(stock_data)

                # 创建技术指标
                self.logger.debug(f"{ts_code}: 开始创建技术指标...")
                featured_data = self.feature_engineer.create_technical_features(processed_data)

                if featured_data is not None:
                    features[ts_code] = featured_data
                    self.logger.debug(f"{ts_code}: 成功创建 {len(featured_data.columns)} 个特征")

            except Exception as e:
                self.logger.error(f"{ts_code} 特征工程失败: {str(e)}")

            progress.update()

        progress.close()
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
            self.logger.info("开始回测...")

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

    def initialize_components(self):
        """初始化系统组件"""
        self.model_ensemble = ModelEnsemble()
        self.feature_selector = FeatureSelector()
        self.data_manager = DataManager()
        self.strategy_evaluator = StrategyEvaluator()

    def train_ensemble_model(self, X_train, y_train, X_val, y_val, original_data):
        """训练集成模型"""
        try:
            # 特征选择
            selected_features = self.feature_selector.select_features(
                X=X_train,
                y=y_train,
                method='random_forest',
                n_features=20
            )

            X_train_selected = X_train[selected_features]
            X_val_selected = X_val[selected_features]

            # 构建和训练模型
            self.model_ensemble.build_ensemble()
            self.model_ensemble.fit(X_train_selected, y_train)

            # 优化权重
            best_weights = self.model_ensemble.optimize_weights(X_val_selected, y_val)

            # 使用最优权重进行预测
            val_predictions = self.model_ensemble.predict(X_val_selected)
            val_score = accuracy_score(y_val, val_predictions)

            return {
                'accuracy': val_score,
                'selected_features': selected_features,
                'validation_predictions': val_predictions,
                'optimized_weights': best_weights,
                'price_data': original_data  # 添加原始价格数据
            }

        except Exception as e:
            logging.error(f"模型训练失败: {str(e)}")
            raise

    def run_realtime_trading(self):
        """运行实时交易"""
        self.logger.info("启动实时交易系统...")

        try:
            while True:
                # 检查市场状态
                if not self.trading_monitor.check_market_status():
                    self.logger.debug("当前不是交易时间，等待...")
                    time.sleep(60)
                    continue
                # 1. 获取实时数据
                self.logger.debug("获取实时行情数据...")
                realtime_data = self.data_engine.get_realtime_data()

                if realtime_data is None or realtime_data.empty:
                    self.logger.warning("未获取到实时数据，等待下次更新...")
                    time.sleep(Config.RETRY_INTERVAL)
                    continue

                # 2. 更新市场状态
                current_prices = dict(zip(realtime_data['ts_code'], realtime_data['close']))
                self.risk_manager.update_portfolio(self.trade_executor.positions, current_prices)

                # 3. 生成特征
                features = self._generate_realtime_features(realtime_data)

                # 4. 生成预测
                predictions = self._generate_predictions(features)

                # 5. 生成交易信号
                signals = self._generate_trading_signals(predictions, realtime_data)

                # 6. 风险检查
                self._check_risk_metrics()

                # 7. 执行交易
                if signals:
                    execution_results = self.trade_executor.execute_trades(signals, current_prices)
                    self._process_execution_results(execution_results)

                # 8. 更新持仓状态
                self._update_portfolio_status()

                # 9. 性能监控
                self.trading_monitor.monitor_performance(
                    self.risk_manager.total_value,
                    self.trade_executor.positions
                )

                # 10. 输出交易状态
                self._print_trading_status()

                # 10. 等待下次更新
                time.sleep(Config.REFRESH_INTERVAL)

        except KeyboardInterrupt:
            self.logger.info("收到终止信号，正在安全退出...")
            self._save_trading_state()
        except Exception as e:
            self.logger.error(f"实时交易系统发生错误: {str(e)}")
            raise

    def monitor_risk(self):
        """风险监控"""
        try:
            # 1. 获取当前持仓
            positions = self.get_current_positions()

            # 2. 计算风险指标
            risk_metrics = self.risk_manager.calculate_risk_metrics(positions)

            # 3. 检查风险限制
            if risk_metrics['portfolio_var'] > Config.MAX_PORTFOLIO_VAR:
                self.reduce_positions()

            if risk_metrics['max_drawdown'] > Config.MAX_DRAWDOWN:
                self.stop_trading()

            # 4. 更新风险报告
            self.risk_manager.update_risk_report(risk_metrics)

        except Exception as e:
            logging.error(f"风险监控发生错误: {str(e)}")
            raise

    def generate_trading_signals(self, predictions):
        """生成交易信号"""
        signals = {}

        for symbol, pred in predictions.items():
            # 1. 检查预测概率
            if pred['buy_prob'] > Config.BUY_THRESHOLD:
                signals[symbol] = 1
            elif pred['sell_prob'] > Config.SELL_THRESHOLD:
                signals[symbol] = -1
            else:
                signals[symbol] = 0

            # 2. 考虑持仓限制
            if signals[symbol] != 0:
                if not self.risk_manager.check_position_limit(symbol):
                    signals[symbol] = 0

            # 3. 检查交易频率
            if not self._check_trade_frequency(symbol):
                signals[symbol] = 0

        return signals

    def _generate_realtime_features(self, realtime_data):
        """生成实时特征"""
        try:
            # 对每只股票生成特征
            features = {}
            for ts_code, data in realtime_data.groupby('ts_code'):
                # 获取历史数据
                hist_data = self.data_engine.get_daily_data(ts_code)
                if hist_data is None or hist_data.empty:
                    continue

                # 合并实时数据
                combined_data = pd.concat([hist_data, data.tail(1)])

                # 生成特征
                stock_features = self.feature_engineer.create_technical_features(combined_data)
                if stock_features is not None:
                    features[ts_code] = stock_features.iloc[-1]  # 只取最新的特征

            return features

        except Exception as e:
            self.logger.error(f"生成实时特征失败: {str(e)}")
            return {}

    def _generate_predictions(self, features):
        """生成预测结果"""
        try:
            predictions = {}
            for ts_code, feature_values in features.items():
                # 使用训练好的模型进行预测
                pred = self.model_engine.predict(feature_values)
                if pred is not None:
                    predictions[ts_code] = {
                        'buy_prob': pred[0],
                        'sell_prob': pred[1] if len(pred) > 1 else 1 - pred[0]
                    }
            return predictions

        except Exception as e:
            self.logger.error(f"生成预测失败: {str(e)}")
            return {}

    def _check_risk_metrics(self):
        """检查风险指标"""
        risk_metrics = self.risk_manager.get_risk_metrics()

        # 检查是否需要调整仓位
        if risk_metrics['concentration_risk'] > Config.MAX_CONCENTRATION:
            adjustments = self.risk_manager.get_position_adjustment()
            if adjustments:
                self.logger.warning("检测到集中度风险，开始调整仓位...")
                self._execute_position_adjustments(adjustments)

    def _process_execution_results(self, results):
        """处理交易执行结果"""
        for symbol, result in results.items():
            if result['status'] == 'success':
                self.logger.info(f"{symbol} 交易执行成功: {result['order']}")
                self._last_trade_time[symbol] = pd.Timestamp.now()
            else:
                self.logger.error(f"{symbol} 交易执行失败: {result['error']}")

    def _update_portfolio_status(self):
        """更新并记录投资组合状态"""
        try:
            portfolio_status = {
                'timestamp': pd.Timestamp.now(),
                'positions': self.trade_executor.positions.copy(),
                'total_value': self.risk_manager.total_value,
                'position_values': self.risk_manager.position_values.copy()
            }

            # 保存状态到数据库或文件
            self._save_portfolio_status(portfolio_status)

        except Exception as e:
            self.logger.error(f"更新投资组合状态失败: {str(e)}")

    def _save_trading_state(self):
        """保存交易状态"""
        try:
            # 保存当前持仓
            self._save_portfolio_status({
                'timestamp': pd.Timestamp.now(),
                'positions': self.trade_executor.positions,
                'orders': self.trade_executor.orders
            })

            self.logger.info("交易状态已保存")

        except Exception as e:
            self.logger.error(f"保存交易状态失败: {str(e)}")

    def _print_trading_status(self):
        """输出交易状态"""
        status = f"""
           ===== 交易状态 =====
           时间: {pd.Timestamp.now()}
           总资产: {self.risk_manager.total_value:.2f}
           持仓数量: {len(self.trade_executor.positions)}
           今日交易: {len([o for o in self.trade_executor.orders if o['timestamp'].date() == pd.Timestamp.now().date()])}
           ==================
           """
        self.logger.info(status)

    def _split_dataset(self, X, y, test_size=0.2, random_state=42):
        """
        划分数据集为训练集和验证集。

        Parameters:
        -----------
        X : DataFrame
            特征数据
        y : Series
            目标变量
        test_size : float
            测试集比例（默认为 0.2）
        random_state : int
            随机种子（默认为 42）

        Returns:
        --------
        dict : 包含训练集和验证集的字典
            {
                'train': {'X': X_train, 'y': y_train},
                'valid': {'X': X_valid, 'y': y_valid}
            }
        """
        from sklearn.model_selection import train_test_split

        try:
            X_train, X_valid, y_train, y_valid = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            return {
                'train': {'X': X_train, 'y': y_train},
                'valid': {'X': X_valid, 'y': y_valid},
            }
        except Exception as e:
            raise RuntimeError(f"数据集划分失败: {str(e)}")

    def get_benchmark_data(self, start_date, end_date):
        """获取沪深300指数数据作为基准"""
        try:
            # 获取沪深300数据 (代码: 000300.SH)
            benchmark_data = self.data_engine.index_daily(
                ts_code='000300.SH',
                start_date=start_date,
                end_date=end_date,
                fields='trade_date,close'
            )
            benchmark_data = benchmark_data.sort_values('trade_date')
            benchmark_returns = benchmark_data['close'].pct_change()
            return benchmark_returns.dropna()
        except Exception as e:
            logging.error(f"获取基准数据失败: {str(e)}")
            return None

    def calculate_returns(self, predictions, price_data, trade_cost=0.0003, slippage=0.002, max_hold_days=10):
        """
        计算策略收益率，加入持仓时间限制和风险控制
        """
        position = 0
        returns = []
        hold_days = 0
        entry_price = 0
        max_loss_per_trade = 0.02  # 单笔最大亏损2%
        profit_target = 0.03  # 目标利润3%


        # 确保数据长度匹配
        n = min(len(predictions), len(price_data))
        predictions = predictions[-n:]
        prices = price_data['close'].values[-n:]

        for i in range(len(predictions)):
            if i == 0:
                returns.append(0)
                continue

            daily_return = (prices[i] / prices[i - 1]) - 1

            # 持仓超过最大天数，强制平仓
            if position == 1 and hold_days >= max_hold_days:
                position = 0
                returns.append(daily_return - trade_cost - slippage)
                hold_days = 0
                continue

            # 计算当前亏损
            if position == 1:
                current_loss = (prices[i] / entry_price - 1)
                # 如果亏损超过3%，强制止损
                if current_loss < -0.03:
                    position = 0
                    returns.append(daily_return - trade_cost - slippage)
                    hold_days = 0
                    continue

            if predictions[i] == 1 and position == 0:
                position = 1
                entry_price = prices[i]
                hold_days = 0
                returns.append(-trade_cost - slippage)
            elif predictions[i] == 2 and position == 1:
                position = 0
                returns.append(daily_return - trade_cost - slippage)
                hold_days = 0
            else:
                if position == 1:
                    hold_days += 1
                    returns.append(daily_return)
                else:
                    returns.append(0)

            if position == 1:
                # 计算当前收益
                current_return = (prices[i] / entry_price - 1)

                # 止损条件
                if current_return < -max_loss_per_trade:
                    position = 0
                    returns.append(-max_loss_per_trade - trade_cost - slippage)
                    continue

                # 止盈条件
                if current_return > profit_target:
                    position = 0
                    returns.append(current_return - trade_cost - slippage)
                    continue

                # 时间止损
                if hold_days >= max_hold_days:
                    position = 0
                    returns.append(current_return - trade_cost - slippage)
                    continue

        return np.array(returns)