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

            # 保存特征名称
            self.feature_names = features.columns.tolist()

            return features

        except Exception as e:
            logging.error(f"特征创建失败: {str(e)}")
            return None

    def _add_price_features(self, df, features):
        """基础价格特征"""
        # 价格变化
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # 日内价格特征
        features['price_range'] = df['high'] - df['low']
        features['price_range_ratio'] = features['price_range'] / df['close']
        features['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        features['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']

        # 开盘跳空
        features['gap'] = df['open'] - df['close'].shift(1)
        features['gap_ratio'] = features['gap'] / df['close'].shift(1)

        # 日内强度
        features['intraday_strength'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        features['open_close_ratio'] = (df['close'] - df['open']) / df['open']

        return features

    def _add_trend_indicators(self, df, features):
        """趋势指标"""
        # EMA
        for period in [5, 10, 20, 30, 60]:
            features[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            features[f'ema_diff_{period}'] = df['close'] - features[f'ema_{period}']
            if period <= 20:  # 只对短期均线计算斜率
                features[f'ema_slope_{period}'] = talib.LINEARREG_SLOPE(features[f'ema_{period}'], timeperiod=5)

        # MACD
        features['macd'], features['macd_signal'], features['macd_hist'] = talib.MACD(
            df['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )

        # DMI
        features['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        features['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        features['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)

        # TRIX
        features['trix'] = talib.TRIX(df['close'], timeperiod=30)

        # TEMA - Triple Exponential Moving Average
        features['tema'] = talib.TEMA(df['close'], timeperiod=20)

        return features

    def _add_volatility_indicators(self, df, features):
        """波动性指标"""
        # ATR及其衍生指标
        features['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        features['atr_ratio'] = features['atr'] / df['close']
        features['natr'] = talib.NATR(df['high'], df['low'], df['close'], timeperiod=14)

        # 历史波动率
        for period in [5, 10, 20]:
            features[f'volatility_{period}'] = df['close'].rolling(window=period).std()
            features[f'volatility_ratio_{period}'] = (
                    features[f'volatility_{period}'] /
                    features[f'volatility_{period}'].rolling(window=period).mean()
            )

        # 布林带
        features['bb_upper'], features['bb_middle'], features['bb_lower'] = talib.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle']

        # KC - Keltner Channel
        features['kc_middle'] = features['ema_20']
        features['kc_upper'] = features['kc_middle'] + 2 * features['atr']
        features['kc_lower'] = features['kc_middle'] - 2 * features['atr']

        return features

    def _add_volume_money_flow_indicators(self, df, features):
        """成交量和资金流向指标"""
        # 成交量
        features['volume_change'] = df['vol'].pct_change()
        for period in [5, 10, 20]:
            features[f'volume_ma_{period}'] = df['vol'].rolling(window=period).mean()
        features['volume_ratio'] = df['vol'] / features['volume_ma_5']
        features['volume_rsi'] = talib.RSI(df['vol'], timeperiod=14)

        # 成交量分布
        features['volume_std_5'] = df['vol'].rolling(window=5).std()
        features['volume_skew_5'] = df['vol'].rolling(window=5).skew()

        # 资金流向
        features['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['vol'], timeperiod=14)
        features['obv'] = talib.OBV(df['close'], df['vol'])
        features['obv_slope'] = talib.LINEARREG_SLOPE(features['obv'], timeperiod=5)

        # 价量相关性
        features['price_volume_corr_5'] = df['close'].rolling(5).corr(df['vol'])
        features['price_volume_corr_10'] = df['close'].rolling(10).corr(df['vol'])

        # 资金流量比率
        features['money_flow_ratio'] = (
                                               (df['close'] - df['low']) - (df['high'] - df['close'])
                                       ) / (df['high'] - df['low'])

        return features

    def _add_momentum_indicators(self, df, features):
        """动量指标"""
        # RSI
        for period in [6, 12, 14, 24]:
            features[f'rsi_{period}'] = talib.RSI(df['close'], timeperiod=period)

        # KDJ
        features['slowk'], features['slowd'] = talib.STOCH(
            df['high'], df['low'], df['close'],
            fastk_period=9, slowk_period=3, slowk_matype=0,
            slowd_period=3, slowd_matype=0
        )

        # ROC - Rate of Change
        for period in [5, 10, 20]:
            features[f'roc_{period}'] = talib.ROC(df['close'], timeperiod=period)

        # CCI - Commodity Channel Index
        features['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)

        # Williams %R
        features['willr'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)

        # Ultimate Oscillator
        features['ultosc'] = talib.ULTOSC(df['high'], df['low'], df['close'],
                                          timeperiod1=7, timeperiod2=14, timeperiod3=28)

        return features

    def _add_pattern_indicators(self, df, features):
        """价格形态指标"""
        # 蜡烛图形态
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

        for pattern_name, pattern_func in pattern_functions.items():
            features[f'pattern_{pattern_name}'] = pattern_func(
                df['open'], df['high'], df['low'], df['close']
            )

        return features

    def _add_support_resistance_indicators(self, df, features):
        """支撑阻力位指标"""
        # 支撑阻力位
        for period in [5, 10, 20]:
            features[f'highest_{period}'] = df['high'].rolling(window=period).max()
            features[f'lowest_{period}'] = df['low'].rolling(window=period).min()

            # 当前价格与支撑阻力位的距离
            features[f'dist_to_resistance_{period}'] = (
                                                               features[f'highest_{period}'] - df['close']
                                                       ) / df['close']
            features[f'dist_to_support_{period}'] = (
                                                            df['close'] - features[f'lowest_{period}']
                                                    ) / df['close']

        # 价格突破指标
        features['breakthrough_up'] = (
                (df['close'] > features['highest_20'].shift(1)) &
                (df['close'].shift(1) <= features['highest_20'].shift(1))
        ).astype(int)

        features['breakthrough_down'] = (
                (df['close'] < features['lowest_20'].shift(1)) &
                (df['close'].shift(1) >= features['lowest_20'].shift(1))
        ).astype(int)

        return features

    def _add_microstructure_indicators(self, df, features):
        """市场微观结构指标"""
        # 价格波动效率
        features['efficiency_ratio'] = abs(df['close'] - df['close'].shift(5)) / (
                df['high'].rolling(5).max() - df['low'].rolling(5).min()
        )

        # 日内价格波动
        features['intraday_volatility'] = (df['high'] - df['low']) / df['open']

        # 价格加速度
        features['price_acceleration'] = features['returns'] - features['returns'].shift(1)

        # 成交量加速度
        features['volume_acceleration'] = features['volume_change'] - features['volume_change'].shift(1)

        return features

    def _add_limit_move_indicators(self, df, features):
        """涨跌停和大幅波动指标"""
        # 假设涨跌停幅度为10%
        LIMIT_PCT = 0.10

        # 涨跌停标志
        features['up_limit'] = (
                (df['close'] >= df['close'].shift(1) * (1 + LIMIT_PCT)) &
                (df['high'] == df['low'])
        ).astype(int)

        features['down_limit'] = (
                (df['close'] <= df['close'].shift(1) * (1 - LIMIT_PCT)) &
                (df['high'] == df['low'])
        ).astype(int)

        # 大幅波动
        features['large_fluctuation'] = (
                abs(df['close'] / df['close'].shift(1) - 1) > LIMIT_PCT * 0.8
        ).astype(int)

        # 连续涨跌停计数
        features['consecutive_up_limit'] = features['up_limit'].rolling(
            window=5, min_periods=1
        ).sum()
        features['consecutive_down_limit'] = features['down_limit'].rolling(
            window=5, min_periods=1
        ).sum()

        return features