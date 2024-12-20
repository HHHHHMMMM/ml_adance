import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import logging


class TradingSignalGenerator:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def add_rule_based_label(self, df, close_column='close'):
        """
        基于规则的交易信号生成
        0: 持有, 1: 买入, 2: 卖出
        """
        try:
            required_features = [
                'ema_5', 'ema_10', 'ema_20', 'macd_hist', 'rsi_14',
                'atr_ratio', 'volume_ratio', 'mfi', 'cci', 'bb_width',
                'price_volume_corr_5', 'efficiency_ratio', 'intraday_volatility'
            ]

            if not all(col in df.columns for col in required_features):
                raise ValueError(f"数据缺少必要特征: {required_features}")

            df = df.copy()

            # 趋势和动量条件
            trend_condition = (
                    (df['ema_5'] > df['ema_10']) &
                    (df['ema_10'] > df['ema_20']) &
                    (df['macd_hist'] > 0) &
                    (df['macd_hist'] > df['macd_hist'].shift(1))
            )

            # RSI和MFI条件（市场动量和资金流向）
            momentum_condition = (
                    (df['rsi_14'] > 35) & (df['rsi_14'] < 70) &
                    (df['rsi_14'] > df['rsi_14'].shift(1)) &
                    (df['mfi'] > 40) & (df['mfi'] < 80)
            )

            # 波动性和效率条件
            volatility_condition = (
                    (df['atr_ratio'] < df['atr_ratio'].rolling(10).mean()) &
                    (df['bb_width'] < df['bb_width'].rolling(20).mean()) &
                    (df['efficiency_ratio'] > 0.5)
            )

            # 成交量和价量关系条件
            volume_condition = (
                    (df['volume_ratio'] > 1.2) &
                    (df['volume_ratio'] < 3.0) &
                    (df['price_volume_corr_5'] > 0.3)
            )

            # 价格形态和突破条件
            pattern_condition = (
                    (df['pattern_engulfing'] > 0) |
                    (df['pattern_morning_star'] > 0) |
                    (df['breakthrough_up'] > 0)
            )

            # 买入条件整合
            buy_condition = (
                    trend_condition &
                    momentum_condition &
                    volatility_condition &
                    volume_condition &
                    (df['intraday_volatility'] < df['intraday_volatility'].rolling(10).mean())
            )

            # 卖出条件
            sell_condition = (
                # 趋势反转
                    ((df['ema_5'] < df['ema_10']) &
                     (df['ema_10'] < df['ema_20']) &
                     (df['macd_hist'] < 0)) |

                    # RSI超买
                    (df['rsi_14'] > 80) |

                    # 价量背离
                    ((df['close'] > df['close'].shift(1)) &
                     (df['volume_ratio'] < 0.7)) |

                    # 形态反转信号
                    (df['pattern_evening_star'] > 0) |
                    (df['pattern_shooting_star'] > 0) |

                    # 突破支撑位
                    (df['breakthrough_down'] > 0)
            )

            # 设置标签
            df['action'] = 0  # 默认持有
            df.loc[buy_condition, 'action'] = 1  # 买入信号
            df.loc[sell_condition, 'action'] = 2  # 卖出信号

            # 避免连续信号
            df.loc[(df['action'] == 1) & (df['action'].shift(1) == 1), 'action'] = 0
            df.loc[(df['action'] == 2) & (df['action'].shift(1) == 2), 'action'] = 0

            return df

        except Exception as e:
            logging.error(f"标签生成失败: {str(e)}")
            return None

    def generate_ml_labels(self, df, forward_period=1, return_threshold=0.02):
        """
        基于未来收益生成机器学习标签
        1: 买入 (未来收益 > threshold)
        0: 不操作 (-threshold <= 未来收益 <= threshold)
        2: 卖出 (未来收益 < -threshold)
        """
        df = df.copy()

        # 计算未来收益
        future_returns = df['close'].shift(-forward_period) / df['close'] - 1

        # 生成标签
        df['ml_label'] = 0
        df.loc[future_returns > return_threshold, 'ml_label'] = 1
        df.loc[future_returns < -return_threshold, 'ml_label'] = 2

        return df

    def train_ml_model(self, features_df, look_forward_days=1, return_threshold=0.02):
        """
        训练机器学习模型预测交易信号
        """
        try:
            # 准备特征和标签
            features_df = self.generate_ml_labels(features_df,
                                                  forward_period=look_forward_days,
                                                  return_threshold=return_threshold)

            # 使用所有量化指标作为特征
            exclude_cols = ['ml_label', 'action', 'open', 'high', 'low', 'close',
                            'vol', 'amount', 'up_limit', 'down_limit']
            feature_cols = [col for col in features_df.columns
                            if col not in exclude_cols]

            X = features_df[feature_cols].copy()
            y = features_df['ml_label'].copy()

            # 标准化特征
            X_scaled = self.scaler.fit_transform(X)

            # 训练随机森林模型
            self.model = RandomForestClassifier(n_estimators=100,
                                                max_depth=10,
                                                min_samples_split=20,
                                                min_samples_leaf=10,
                                                random_state=42)
            self.model.fit(X_scaled, y)

            # 获取特征重要性
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            logging.info("模型训练完成")
            logging.info("\n最重要的10个特征:\n{}".format(
                feature_importance.head(10)))

            return True

        except Exception as e:
            logging.error(f"模型训练失败: {str(e)}")
            return False

    def predict_signals(self, features_df):
        """
        使用训练好的模型预测交易信号
        """
        try:
            if self.model is None:
                raise ValueError("模型未训练，请先调用train_ml_model()")

            # 准备特征
            feature_cols = [col for col in features_df.columns
                            if col not in ['ml_label', 'action', 'close', 'open', 'high', 'low']]
            X = features_df[feature_cols].copy()

            # 标准化特征
            X_scaled = self.scaler.transform(X)

            # 预测
            predictions = self.model.predict(X_scaled)

            # 添加预测结果
            features_df['ml_prediction'] = predictions

            return features_df

        except Exception as e:
            logging.error(f"预测失败: {str(e)}")
            return None


def combine_signals(df, alpha=0.7):
    """
    合并规则基和机器学习的信号
    alpha: 机器学习信号的权重
    """
    df = df.copy()

    # 确保两种信号都存在
    if 'action' not in df.columns or 'ml_prediction' not in df.columns:
        raise ValueError("缺少必要的信号列")

    # 将规则基信号和机器学习信号结合
    df['final_signal'] = np.where(
        df['action'] == df['ml_prediction'],
        df['action'],
        np.where(
            df['ml_prediction'] == 0,
            df['action'],
            df['ml_prediction'] if np.random.random() < alpha else df['action']
        )
    )

    return df