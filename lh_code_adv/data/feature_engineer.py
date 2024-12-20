# data/feature_engineer.py

import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import StandardScaler
import logging


class FeatureEngineer:
    def __init__(self):
        self.feature_names = None

    def create_features(self, df):
        """
        创建特征
        包括：基础价格特征、趋势指标、波动性指标、成交量指标、动量指标、
             价格形态特征、资金流向指标、情绪指标等
        """
        try:
            features = pd.DataFrame(index=df.index)

            # 基础数据
            essential_columns = ['open', 'high', 'low', 'close', 'vol', 'amount']
            for col in essential_columns:
                if col in df.columns:
                    features[col] = df[col]

            # 1. 基础价格特征
            features = self._add_price_features(df, features)

            # 2. 趋势指标
            features = self._add_trend_indicators(df, features)

            # 3. 波动性指标
            features = self._add_volatility_indicators(df, features)

            # 4. 成交量和资金流向指标
            features = self._add_volume_money_flow_indicators(df, features)

            # 5. 动量指标
            features = self._add_momentum_indicators(df, features)

            # 6. 价格形态指标
            features = self._add_pattern_indicators(df, features)

            # 7. 支撑阻力位指标
            features = self._add_support_resistance_indicators(df, features)

            # 8. 市场微观结构指标
            features = self._add_microstructure_indicators(df, features)

            # 9. 涨跌停和大幅波动指标
            features = self._add_limit_move_indicators(df, features)

            # 去除NaN值
            features = features.dropna()

            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.fillna(method='ffill').fillna(method='bfill')

            # 处理极端值
            for col in features.columns:
                if features[col].dtype in [np.float64, np.float32]:
                    Q1 = features[col].quantile(0.25)
                    Q3 = features[col].quantile(0.75)
                    IQR = Q3 - Q1
                    features[col] = features[col].clip(
                        lower=Q1 - 3 * IQR,
                        upper=Q3 + 3 * IQR
                    )
            # 保存特征名称
            self.feature_names = features.columns.tolist()

            return features

        except Exception as e:
            logging.error(f"特征创建失败: {str(e)}")
            return None

    def _add_price_features(self, df, features):
        """基础价格特征"""
        feature_dict = {
            'returns': df['close'].pct_change(),
            'log_returns': np.log(df['close'] / df['close'].shift(1)),
            'price_range': df['high'] - df['low'],
            'upper_shadow': df['high'] - np.maximum(df['open'], df['close']),
            'lower_shadow': np.minimum(df['open'], df['close']) - df['low'],
            'gap': df['open'] - df['close'].shift(1),
            'intraday_strength': (df['close'] - df['low']) / (df['high'] - df['low']),
            'open_close_ratio': (df['close'] - df['open']) / df['open']
        }

        # 依赖其他特征的计算
        feature_dict['price_range_ratio'] = feature_dict['price_range'] / df['close']
        feature_dict['gap_ratio'] = feature_dict['gap'] / df['close'].shift(1)

        return pd.concat([features, pd.DataFrame(feature_dict, index=df.index)], axis=1)

    def _add_trend_indicators(self, df, features):
        """趋势指标"""
        feature_dict = {}

        # EMA相关特征
        for period in [5, 10, 20, 30, 60]:
            ema = df['close'].ewm(span=period, adjust=False).mean()
            feature_dict[f'ema_{period}'] = ema
            feature_dict[f'ema_diff_{period}'] = df['close'] - ema
            if period <= 20:
                feature_dict[f'ema_slope_{period}'] = talib.LINEARREG_SLOPE(ema, timeperiod=5)

        # MACD
        macd, signal, hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        feature_dict.update({
            'macd': macd,
            'macd_signal': signal,
            'macd_hist': hist,
            'plus_di': talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14),
            'minus_di': talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14),
            'adx': talib.ADX(df['high'], df['low'], df['close'], timeperiod=14),
            'trix': talib.TRIX(df['close'], timeperiod=30),
            'tema': talib.TEMA(df['close'], timeperiod=20)
        })

        return pd.concat([features, pd.DataFrame(feature_dict, index=df.index)], axis=1)

    def _add_volatility_indicators(self, df, features):
        """波动性指标"""
        feature_dict = {
            'atr': talib.ATR(df['high'], df['low'], df['close'], timeperiod=14),
            'natr': talib.NATR(df['high'], df['low'], df['close'], timeperiod=14)
        }

        # 添加ATR比率
        feature_dict['atr_ratio'] = feature_dict['atr'] / df['close']

        # 波动率特征
        for period in [5, 10, 20]:
            vol = df['close'].rolling(window=period).std()
            feature_dict[f'volatility_{period}'] = vol
            feature_dict[f'volatility_ratio_{period}'] = (
                    vol / vol.rolling(window=period).mean()
            )

        # 布林带
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        feature_dict.update({
            'bb_upper': bb_upper,
            'bb_middle': bb_middle,
            'bb_lower': bb_lower,
            'bb_width': (bb_upper - bb_lower) / bb_middle
        })

        # Keltner Channel
        ema_20 = df['close'].ewm(span=20, adjust=False).mean()
        atr = feature_dict['atr']
        feature_dict.update({
            'kc_middle': ema_20,
            'kc_upper': ema_20 + 2 * atr,
            'kc_lower': ema_20 - 2 * atr
        })

        return pd.concat([features, pd.DataFrame(feature_dict, index=df.index)], axis=1)

    def _add_volume_money_flow_indicators(self, df, features):
        """成交量和资金流向指标"""
        feature_dict = {
            'volume_change': df['vol'].pct_change()
        }

        # 成交量均线
        for period in [5, 10, 20]:
            feature_dict[f'volume_ma_{period}'] = df['vol'].rolling(window=period).mean()

        # 成交量相关指标
        feature_dict.update({
            'volume_ratio': df['vol'] / df['vol'].rolling(window=5).mean(),
            'volume_rsi': talib.RSI(df['vol'], timeperiod=14),
            'volume_std_5': df['vol'].rolling(window=5).std(),
            'volume_skew_5': df['vol'].rolling(window=5).skew(),
            'mfi': talib.MFI(df['high'], df['low'], df['close'], df['vol'], timeperiod=14),
            'obv': talib.OBV(df['close'], df['vol'])
        })

        # OBV斜率
        feature_dict['obv_slope'] = talib.LINEARREG_SLOPE(feature_dict['obv'], timeperiod=5)

        # 价量相关性
        feature_dict.update({
            'price_volume_corr_5': df['close'].rolling(5).corr(df['vol']),
            'price_volume_corr_10': df['close'].rolling(10).corr(df['vol']),
            'money_flow_ratio': ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        })

        return pd.concat([features, pd.DataFrame(feature_dict, index=df.index)], axis=1)

    def _add_momentum_indicators(self, df, features):
        """动量指标"""
        feature_dict = {}

        # RSI
        for period in [6, 12, 14, 24]:
            feature_dict[f'rsi_{period}'] = talib.RSI(df['close'], timeperiod=period)

        # KDJ
        slowk, slowd = talib.STOCH(
            df['high'], df['low'], df['close'],
            fastk_period=9, slowk_period=3, slowk_matype=0,
            slowd_period=3, slowd_matype=0
        )
        feature_dict.update({
            'slowk': slowk,
            'slowd': slowd
        })

        # ROC
        for period in [5, 10, 20]:
            feature_dict[f'roc_{period}'] = talib.ROC(df['close'], timeperiod=period)

        # 其他动量指标
        feature_dict.update({
            'cci': talib.CCI(df['high'], df['low'], df['close'], timeperiod=14),
            'willr': talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14),
            'ultosc': talib.ULTOSC(df['high'], df['low'], df['close'],
                                   timeperiod1=7, timeperiod2=14, timeperiod3=28)
        })

        return pd.concat([features, pd.DataFrame(feature_dict, index=df.index)], axis=1)

    def _add_pattern_indicators(self, df, features):
        """价格形态指标"""
        pattern_functions = {
            'doji': talib.CDLDOJI,
            'hammer': talib.CDLHAMMER,
            'hanging_man': talib.CDLHANGINGMAN,
            'engulfing': talib.CDLENGULFING,
            'morning_star': talib.CDLMORNINGSTAR,
            'evening_star': talib.CDLEVENINGSTAR,
            'shooting_star': talib.CDLSHOOTINGSTAR,
            'three_white_soldiers': talib.CDL3WHITESOLDIERS,
            'three_black_crows': talib.CDL3BLACKCROWS
        }

        feature_dict = {
            f'pattern_{pattern_name}': pattern_func(df['open'], df['high'], df['low'], df['close'])
            for pattern_name, pattern_func in pattern_functions.items()
        }

        return pd.concat([features, pd.DataFrame(feature_dict, index=df.index)], axis=1)

    def _add_support_resistance_indicators(self, df, features):
        """支撑阻力位指标"""
        feature_dict = {}

        # 计算支撑阻力位
        for period in [5, 10, 20]:
            highest = df['high'].rolling(window=period).max()
            lowest = df['low'].rolling(window=period).min()

            feature_dict.update({
                f'highest_{period}': highest,
                f'lowest_{period}': lowest,
                f'dist_to_resistance_{period}': (highest - df['close']) / df['close'],
                f'dist_to_support_{period}': (df['close'] - lowest) / df['close']
            })

        # 突破指标
        feature_dict.update({
            'breakthrough_up': (
                    (df['close'] > feature_dict['highest_20'].shift(1)) &
                    (df['close'].shift(1) <= feature_dict['highest_20'].shift(1))
            ).astype(int),

            'breakthrough_down': (
                    (df['close'] < feature_dict['lowest_20'].shift(1)) &
                    (df['close'].shift(1) >= feature_dict['lowest_20'].shift(1))
            ).astype(int)
        })

        return pd.concat([features, pd.DataFrame(feature_dict, index=df.index)], axis=1)

    def _add_microstructure_indicators(self, df, features):
        """市场微观结构指标"""
        returns = df['close'].pct_change()
        volume_change = df['vol'].pct_change()

        feature_dict = {
            'efficiency_ratio': abs(df['close'] - df['close'].shift(5)) / (
                    df['high'].rolling(5).max() - df['low'].rolling(5).min()
            ),
            'intraday_volatility': (df['high'] - df['low']) / df['open'],
            'price_acceleration': returns - returns.shift(1),
            'volume_acceleration': volume_change - volume_change.shift(1)
        }

        return pd.concat([features, pd.DataFrame(feature_dict, index=df.index)], axis=1)
    def _add_limit_move_indicators(self, df, features):
        """涨跌停和大幅波动指标"""
        # 假设涨跌停幅度为10%
        LIMIT_PCT = 0.10

        # 预先计算所有特征，存储在字典中
        feature_dict = {
            'up_limit': (
                    (df['close'] >= df['close'].shift(1) * (1 + LIMIT_PCT)) &
                    (df['high'] == df['low'])
            ).astype(int),

            'down_limit': (
                    (df['close'] <= df['close'].shift(1) * (1 - LIMIT_PCT)) &
                    (df['high'] == df['low'])
            ).astype(int),

            'large_fluctuation': (
                    abs(df['close'] / df['close'].shift(1) - 1) > LIMIT_PCT * 0.8
            ).astype(int)
        }

        # 添加连续涨跌停计数
        feature_dict['consecutive_up_limit'] = feature_dict['up_limit'].rolling(
            window=5, min_periods=1
        ).sum()
        feature_dict['consecutive_down_limit'] = feature_dict['down_limit'].rolling(
            window=5, min_periods=1
        ).sum()

        # 一次性将所有特征添加到 features DataFrame
        return pd.concat([features, pd.DataFrame(feature_dict, index=df.index)], axis=1)