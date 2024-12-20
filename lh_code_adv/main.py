import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

from lh_code_adv.QuantTrading import QuantTradingSystem
from lh_code_adv.backtesting.backtesting_engine import Backtester
from lh_code_adv.trading.TradingSignalGenerator import TradingSignalGenerator
from utils.logger import setup_logger
import logging

pd.set_option('display.max_columns', None)

def main():
    # 设置日志记录器并获取它
    logger = setup_logger('quant_trading')

    # 初始化系统
    logger.info("初始化量化交易系统...")
    trading_system = QuantTradingSystem()
    trading_system.initialize_components()
    backtesting_system=Backtester()

    try:
        logger.info("开始模型训练阶段...")

        # 1. 获取股票列表
        logger.info("Step 1: 正在获取股票列表...")
        stock_list = trading_system.data_engine.get_stock_list()
        if stock_list is None or len(stock_list) == 0:
            logger.error("获取股票列表失败")
            return
        logger.info(f"成功获取股票列表，共 {len(stock_list)} 只股票")

        # 2. 获取历史数据
        logger.info("Step 2: 开始获取历史数据...")
        historical_data = {}
        i=0
        for ts_code in stock_list['ts_code']:
            i=i+1
            if i==10:
                break
            logger.info(f"正在获取 {ts_code} 的历史数据...")
            data = trading_system.data_engine.get_daily_data(ts_code)
            if data is not None and not data.empty:
                historical_data[ts_code] = data
                logger.info(f"成功获取 {ts_code} 的历史数据，共 {len(data)} 条记录")
            else:
                logger.warning(f"获取 {ts_code} 的历史数据失败")

        if not historical_data:
            logger.error("没有成功获取任何历史数据")
            return
        original_data = historical_data.copy()

        logger.info(f"历史数据获取完成，共获取 {len(historical_data)} 只股票的数据")

        # 3. 数据处理和特征工程
        logger.info("Step 3: 开始数据处理和特征工程...")
        processed_data = {}
        signal_generator = TradingSignalGenerator()

        for ts_code, data in historical_data.items():
            try:
                logger.info(f"正在处理 {ts_code} 的数据...")
                # 数据预处理
                processed = trading_system.data_processor.process_daily_data(data)
                if processed is not None:
                    # 特征工程
                    features = trading_system.feature_engineer.create_features(processed)

                    # 确保保留原始的价格列
                    essential_columns = ['close', 'open', 'high', 'low', 'vol', 'amount']
                    for col in essential_columns:
                        if col not in features.columns and col in processed.columns:
                            features[col] = processed[col]

                    # 生成规则基标签
                    features = signal_generator.add_rule_based_label(features)

                    # 生成机器学习标签（基于未来收益）
                    features = signal_generator.generate_ml_labels(
                        features,
                        forward_period=1,
                        return_threshold=0.02
                    )

                    if 'action' not in features.columns or features['action'].isna().all():
                        logger.warning(f"{ts_code} 的数据无法生成有效的规则基标签")
                        continue

                    if 'ml_label' not in features.columns or features['ml_label'].isna().all():
                        logger.warning(f"{ts_code} 的数据无法生成有效的机器学习标签")
                        continue

                    processed_data[ts_code] = features
                    logger.info(f"成功处理 {ts_code} 的数据，生成 {len(features.columns)} 个特征和两种标签")
            except Exception as e:
                logger.error(f"处理 {ts_code} 的数据失败: {str(e)}")
                continue

        if not processed_data:
            logger.error("没有成功处理任何数据")
            return

        logger.info(f"数据处理完成，共处理 {len(processed_data)} 只股票的数据")

        # 4. 特征选择
        logger.info("Step 4: 开始特征选择...")
        try:
            # 分别为规则基和机器学习准备数据
            price_cols = ['open', 'high', 'low', 'close']  # 这些可以去除
            trading_cols = ['vol', 'amount']  # 这些建议保留
            exclude_cols = ['action', 'ml_label'] + price_cols  # 只去除价格列和标签列

            combined_X = pd.concat([processed_data[ts_code].drop(columns=exclude_cols)
                                    for ts_code in processed_data])

            # 规则基标签
            combined_y_rule = pd.concat([processed_data[ts_code]['action']
                                         for ts_code in processed_data])

            # 机器学习标签
            combined_y_ml = pd.concat([processed_data[ts_code]['ml_label']
                                       for ts_code in processed_data])

            # 清理特征数据
            combined_X = clean_features(combined_X)

            # 检查数据
            logger.info("检查数据状态:")
            logger.info(f"特征数据形状: {combined_X.shape}")
            logger.info(f"是否存在无穷值: {np.any(np.isinf(combined_X))}")
            logger.info(f"是否存在NaN值: {combined_X.isna().any().any()}")

            # 特征选择
            selected_features = comprehensive_feature_selection(
                combined_X,
                combined_y_ml,
                combined_y_rule
            )
            logger.info(f"特征选择完成...")

        except Exception as e:
            logger.error(f"特征选择失败: {str(e)}")
            import traceback
            traceback.print_exc()  # 添加更详细的错误信息
            return

        # 5. 数据集划分
        logger.info("Step 5: 开始划分训练集和验证集...")
        try:
            # 解包特征选择的结果
            feature_names, feature_importance = selected_features  # 从返回的元组中取出特征名称列表

            # 为机器学习模型划分数据集
            dataset_ml = trading_system.time_series_split(
                combined_X[feature_names],  # 只使用特征名称列表
                combined_y_ml
            )
            train_data_ml, valid_data_ml = dataset_ml['train'], dataset_ml['valid']

            logger.info(f"数据集划分完成，训练集样本数: {len(train_data_ml['X'])}, "
                        f"验证集样本数: {len(valid_data_ml['X'])}")
        except Exception as e:
            logger.error(f"数据集划分失败: {str(e)}")
            return

        # 6. 训练模型
        # 修改位置：从第6步开始

        # 6. 训练模型
        logger.info("Step 6: 开始训练模型...")
        try:
            # 训练机器学习模型
            model_performance = trading_system.train_ensemble_model(
                combined_X[feature_names],  # 使用全量特征数据
                combined_y_ml,  # 使用全量标签数据
                None,  # 不需要验证集X
                None,  # 不需要验证集y
                original_data  # 原始数据
            )

            # 获取全量数据的预测结果
            ml_predictions = model_performance['predictions']  # 注意这里改为predictions而不是validation_predictions

            # 生成规则基信号
            rule_predictions = signal_generator.add_rule_based_label(processed_data[list(processed_data.keys())[0]])[
                'action']

            # 确保预测结果长度一致
            min_len = min(len(ml_predictions), len(rule_predictions))
            ml_predictions = ml_predictions[-min_len:]
            rule_predictions = rule_predictions[-min_len:]

            # 合并两种信号
            final_predictions = []
            for ml_pred, rule_pred in zip(ml_predictions, rule_predictions):
                if ml_pred == rule_pred:
                    final_predictions.append(ml_pred)
                elif ml_pred == 0:
                    final_predictions.append(rule_pred)
                else:
                    # 使用alpha权重
                    final_predictions.append(ml_pred if np.random.random() < 0.7 else rule_pred)

            logger.info("模型训练完成")
            logger.info("模型性能指标:")
            for metric, value in model_performance.items():
                logger.info(f"{metric}: {value}")

        except Exception as e:
            logger.error(f"模型训练失败: {str(e)}")
            return

        # 分析信号分布
        signal_distribution_ml = pd.Series(ml_predictions).value_counts()
        signal_distribution_rule = pd.Series(rule_predictions).value_counts()
        signal_distribution_final = pd.Series(final_predictions).value_counts()

        logger.info("信号分布:")
        logger.info(f"\n机器学习信号:\n{signal_distribution_ml}")
        logger.info(f"\n规则基信号:\n{signal_distribution_rule}")
        logger.info(f"\n最终信号:\n{signal_distribution_final}")

        # 7. 评估阶段
        logger.info("Step 7: 开始模型评估...")
        try:
            # 获取股票数据
            stock_code = list(processed_data.keys())[0]
            price_data = historical_data[stock_code]
            price_data = price_data.sort_values('trade_date')

            # 设置回测参数
            backtest_params = {
                'trade_cost': 0.0003,
                'slippage': 0.002,
                'max_hold_days': 10,
                'max_position_size': 0.2,
                'max_loss_per_trade': 0.02,
                'profit_target': 0.03
            }

            # 初始化回测器

            # 分别进行回测
            backtest_results_ml = backtesting_system.backtest(
                predictions=ml_predictions,
                price_data=price_data,
                params=backtest_params
            )

            backtest_results_rule = backtesting_system.backtest(
                predictions=rule_predictions,
                price_data=price_data,
                params=backtest_params
            )

            backtest_results_final = backtesting_system.backtest(
                predictions=final_predictions,
                price_data=price_data,
                params=backtest_params
            )

            # 获取基准收益率
            benchmark_returns = trading_system.get_benchmark_data(
                start_date=price_data['trade_date'].min(),
                end_date=price_data['trade_date'].max()
            )

            # 评估各种策略
            evaluation_results = {
                'ML_Strategy': trading_system.strategy_evaluator.evaluate_strategy(
                    backtest_results_ml['returns'], benchmark_returns
                ),
                'Rule_Strategy': trading_system.strategy_evaluator.evaluate_strategy(
                    backtest_results_rule['returns'], benchmark_returns
                ),
                'Combined_Strategy': trading_system.strategy_evaluator.evaluate_strategy(
                    backtest_results_final['returns'], benchmark_returns
                )
            }

            # 输出详细的回测结果
            logger.info("回测结果:")
            for strategy_name in ['ML_Strategy', 'Rule_Strategy', 'Combined_Strategy']:
                logger.info(f"\n{strategy_name} 回测结果:")
                if strategy_name == 'ML_Strategy':
                    results = backtest_results_ml
                elif strategy_name == 'Rule_Strategy':
                    results = backtest_results_rule
                else:
                    results = backtest_results_final

                logger.info(f"总交易次数: {results['trade_statistics']['total_trades']}")
                logger.info(f"胜率: {results['trade_statistics']['win_rate']:.2%}")
                logger.info(f"平均持仓天数: {results['trade_statistics']['avg_hold_days']:.2f}")
                logger.info(f"最大单笔收益: {results['trade_statistics']['max_profit']:.2%}")
                logger.info(f"最大单笔亏损: {results['trade_statistics']['max_loss']:.2%}")
                logger.info(f"收益因子: {results['trade_statistics']['profit_factor']:.2f}")
                logger.info(f"最终资金: {results['final_capital']:.2f}")
                logger.info(f"总收益率: {results['return_rate']:.2%}")

            logger.info("\n策略评估结果:")
            for strategy_name, results in evaluation_results.items():
                logger.info(f"\n{strategy_name}:")
                for category, metrics in results.items():
                    logger.info(f"\n{category.upper()} METRICS:")
                    for metric, value in metrics.items():
                        logger.info(f"{metric}: {value:.4f}")

        except Exception as e:
            logger.error(f"模型评估失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return

        # 8. 启动实时交易
        logger.info("Step 8: 开始启动实时交易系统...")
        try:
            trading_system.run_realtime_trading()
        except Exception as e:
            logger.error(f"实时交易系统运行错误: {str(e)}")
            return

    except Exception as e:
        logger.error(f"系统运行错误: {str(e)}")
        raise
    finally:
        logger.info("量化交易系统运行结束")

        # 4. 特征选择


def clean_features(combined_X):
    logger = setup_logger('quant_trading')
    """
    清理特征数据中的异常值
    """
    logger.info("开始清理特征数据...")

    # 1. 替换无穷值
    combined_X = combined_X.replace([np.inf, -np.inf], np.nan)

    # 2. 处理过大的值
    # 对每列使用中位数和IQR方法处理异常值
    for col in combined_X.columns:
        Q1 = combined_X[col].quantile(0.25)
        Q3 = combined_X[col].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 3 * IQR
        lower_bound = Q1 - 3 * IQR

        # 将超出范围的值设为边界值
        combined_X[col] = combined_X[col].clip(lower=lower_bound, upper=upper_bound)

    # 3. 填充剩余的NaN值
    combined_X = combined_X.fillna(method='ffill').fillna(method='bfill')

    # 4. 确保所有值都在float64范围内
    for col in combined_X.columns:
        max_val = np.finfo(np.float64).max / 1e10  # 留出安全余量
        min_val = np.finfo(np.float64).min / 1e10
        combined_X[col] = combined_X[col].clip(lower=min_val, upper=max_val)

    logger.info(f"特征数据清理完成，共处理 {len(combined_X.columns)} 个特征")
    return combined_X


def add_action_label(df, close_column='close', volume_column='vol', high_column='high', low_column='low', n_days=1):
    """
    优化的短线交易标签生成
    """
    required_columns = [close_column, volume_column, high_column, low_column]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"数据缺少必要列，请确保包含: {required_columns}")

    df = df.copy()

    # 1. 趋势指标
    # EMA和MACD
    df['EMA5'] = df[close_column].ewm(span=5, adjust=False).mean()
    df['EMA10'] = df[close_column].ewm(span=10, adjust=False).mean()
    df['EMA20'] = df[close_column].ewm(span=20, adjust=False).mean()

    df['MACD'] = df[close_column].ewm(span=12, adjust=False).mean() - df[close_column].ewm(span=26, adjust=False).mean()
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    # 2. 动量指标
    df['RSI'] = calculate_rsi(df[close_column], window=14)

    # 3. 波动率指标
    df['ATR'] = calculate_atr(df[high_column], df[low_column], df[close_column], window=14)
    df['ATR_ratio'] = df['ATR'] / df[close_column]

    # 4. 成交量分析
    df['Volume_MA5'] = df[volume_column].rolling(window=5).mean()
    df['Volume_MA10'] = df[volume_column].rolling(window=10).mean()
    df['Volume_ratio'] = df[volume_column] / df['Volume_MA5']

    # 5. 价格趋势强度
    df['Trend_strength'] = (df['EMA5'] - df['EMA20']) / df['EMA20']
    df['Price_momentum'] = df[close_column].pct_change(3)

    # 6. 止损止盈参考价位
    df['High_5d'] = df[high_column].rolling(window=5).max()
    df['Low_5d'] = df[low_column].rolling(window=5).min()
    df['Stop_loss'] = df[close_column] * 0.97  # 3%止损
    df['Take_profit'] = df[close_column] * 1.05  # 5%止盈

    #7.市场环境判断
    df['Market_Trend'] = (df['EMA20'] > df['EMA20'].shift(20))  # 判断中期趋势
    df['Market_Strength'] = df['EMA5'].rolling(20).mean() / df['EMA20'] - 1  # 市场强度

    # # 买入条件（需要满足多个条件）
    # buy_condition = (
    #     # 趋势向上
    #         (df['EMA5'] > df['EMA10']) &
    #         (df['EMA10'] > df['EMA20']) &
    #
    #         # MACD金叉或柱状图向上
    #         ((df['MACD_hist'] > 0) & (df['MACD_hist'].shift(1) < 0) |
    #          (df['MACD_hist'] > df['MACD_hist'].shift(1))) &
    #
    #         # RSI适中且向上
    #         (df['RSI'] > 30) & (df['RSI'] < 70) &
    #         (df['RSI'] > df['RSI'].shift(1)) &
    #
    #         # 放量上涨
    #         (df['Volume_ratio'] > 1.2) &
    #
    #         # 价格动量为正
    #         (df['Price_momentum'] > 0) &
    #
    #         # 波动率适中
    #         (df['ATR_ratio'] < df['ATR_ratio'].rolling(window=20).mean())
    # )
    #
    # # 卖出条件（满足任一条件）
    # sell_condition = (
    #     # 趋势反转
    #         ((df['EMA5'] < df['EMA10']) & (df['EMA5'].shift(1) > df['EMA10'].shift(1))) |
    #
    #         # MACD死叉
    #         ((df['MACD_hist'] < 0) & (df['MACD_hist'].shift(1) > 0)) |
    #
    #         # RSI超买
    #         (df['RSI'] > 75) |
    #
    #         # 价格跌破支撑
    #         (df[close_column] < df['Low_5d'].shift(1)) |
    #
    #         # 成交量萎缩
    #         (df['Volume_ratio'] < 0.7)
    # )
    #
    # # 设置标签
    # df['action'] = 0  # 默认持有
    # df.loc[buy_condition, 'action'] = 1  # 买入信号
    # df.loc[sell_condition, 'action'] = 2  # 卖出信号
    buy_condition = (
        # 趋势确认（保持原有条件）
            (df['EMA5'] > df['EMA10']) &

            # 动量确认（修改阈值）
            (df['Price_momentum'] > 0.01) &  # 提高动量要求

            # 成交量确认（添加连续性）
            (df['Volume_ratio'] > 1.1) &
            (df['Volume_ratio'].rolling(3).mean() > 1.0) &

            # RSI条件优化
            (df['RSI'] > 35) & (df['RSI'] < 60) &  # 更保守的RSI区间
            (df['RSI'] > df['RSI'].shift(1)) &  # RSI上升

            # 添加支撑位确认
            (df[close_column] > df['Low_5d']) &

            # 波动率确认
            (df['ATR_ratio'] < df['ATR_ratio'].rolling(10).mean())  # 波动率降低
    )

    # 修改卖出条件（更严格）
    sell_condition = (
        # 趋势明确反转
            ((df['EMA5'] < df['EMA10']) &
             (df['EMA10'] < df['EMA20']) &
             (df['MACD_hist'] < 0)) |

            # RSI过高
            (df['RSI'] > 80) |

            # 止损条件
            (df[close_column] < df['Stop_loss']) |

            # 止盈条件
            (df[close_column] > df['Take_profit'])
    )

    # 设置标签
    df['action'] = 0
    df.loc[buy_condition, 'action'] = 1
    df.loc[sell_condition, 'action'] = 2

    df.loc[(df['action'] == 1) & (df['action'].shift(1) == 1), 'action'] = 0
    df.loc[(df['action'] == 2) & (df['action'].shift(1) == 2), 'action'] = 0

    # 记录标签分布
    label_dist = df['action'].value_counts()
    logging.info(f"优化后的标签分布:\n{label_dist}")

    return df.dropna()


def calculate_rsi(close, window=14):
    """计算RSI指标"""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_atr(high, low, close, window=14):
    """计算ATR指标"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    return tr.rolling(window=window).mean()


def comprehensive_feature_selection(X, y_ml, y_rule, method='combined_mi'):
    logger = setup_logger('quant_trading')

    # 创建输入数据的副本以避免碎片化
    X = X.copy()

    if method == 'combined_mi':
        # 计算互信息分数
        mi_scores_ml = mutual_info_classif(X, y_ml)
        mi_scores_rule = mutual_info_classif(X, y_rule)

        # 归一化分数
        mi_scores_ml = mi_scores_ml / np.sum(mi_scores_ml)
        mi_scores_rule = mi_scores_rule / np.sum(mi_scores_rule)

        # 组合分数
        combined_scores = 0.5 * mi_scores_ml + 0.5 * mi_scores_rule

    elif method == 'combined_importance':
        # 使用随机森林
        rf_params = {
            'n_estimators': 100,
            'random_state': 42,
            'n_jobs': -1  # 使用所有CPU核心加速计算
        }
        rf_ml = RandomForestClassifier(**rf_params)
        rf_rule = RandomForestClassifier(**rf_params)

        rf_ml.fit(X, y_ml)
        rf_rule.fit(X, y_rule)

        # 归一化特征重要性
        imp_ml = rf_ml.feature_importances_ / np.sum(rf_ml.feature_importances_)
        imp_rule = rf_rule.feature_importances_ / np.sum(rf_rule.feature_importances_)

        combined_scores = 0.5 * imp_ml + 0.5 * imp_rule

    # 创建特征重要性DataFrame（使用一次性构建而不是逐列添加）
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': combined_scores,
        'importance_ml': mi_scores_ml if method == 'combined_mi' else imp_ml,
        'importance_rule': mi_scores_rule if method == 'combined_mi' else imp_rule
    })

    # 排序
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    # 使用肘部法则选择特征
    n_features = elbow_point(feature_importance['importance'].values)
    selected_features = feature_importance['feature'].iloc[:n_features].tolist()

    # 记录结果
    logger.info(f"特征选择完成，选择了 {len(selected_features)}/{len(X.columns)} 个特征")
    logger.info("\n前10个最重要的特征:")
    logger.info(feature_importance.head(10).to_string())

    return selected_features, feature_importance.to_dict('records')

def elbow_point(values, min_features=5, smooth_window=None):
    logger = setup_logger('quant_trading')

    """
    使用肘部法则确定截断点，带平滑处理

    Parameters:
    -----------
    values : array-like
        特征重要性得分数组（必须是降序排列）
    min_features : int, default=5
        最少保留的特征数量
    smooth_window : int, optional
        平滑窗口大小，用于减少噪声影响
    """
    if len(values) < 3:
        return len(values)

    try:
        values = np.asarray(values, dtype=np.float64)

        # 可选的平滑处理
        if smooth_window:
            kernel = np.ones(smooth_window) / smooth_window
            values = np.convolve(values, kernel, mode='valid')

        # 使用向量化操作代替循环
        x = np.arange(len(values))
        coords = np.column_stack([x, values])

        # 计算首尾连线的单位向量
        line_vec = coords[-1] - coords[0]
        line_length = np.linalg.norm(line_vec)

        if line_length < 1e-10:
            return min_features

        line_vec_norm = line_vec / line_length

        # 向量化计算距离
        vec_from_first = coords - coords[0]
        scalar_proj = np.dot(vec_from_first, line_vec_norm)
        vec_proj = np.outer(scalar_proj, line_vec_norm)
        distances = np.linalg.norm(vec_from_first - vec_proj, axis=1)

        # 找到最优截断点
        elbow_idx = np.argmax(distances) + 1
        return max(elbow_idx, min_features)

    except Exception as e:
        logger.warning(f"肘部法则计算失败：{str(e)}，默认返回{min_features}个特征")
        return min_features

if __name__ == "__main__":
    main()