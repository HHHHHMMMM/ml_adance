# data/feature_engineer.py

import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import StandardScaler
import logging


class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []

    def create_features(self, df):
        """
        创建特征
        包括：技术指标、波动性指标、趋势指标等
        """
        try:
            features = pd.DataFrame(index=df.index)

            essential_columns = ['open', 'high', 'low', 'close', 'vol', 'amount']
            for col in essential_columns:
                if col in df.columns:
                    features[col] = df[col]

            # 基础价格特征
            features = self._add_price_features(df, features)

            # 技术指标
            features = self._add_technical_indicators(df, features)

            # 波动性指标
            features = self._add_volatility_indicators(df, features)

            # 趋势特征
            features = self._add_trend_features(df, features)

            # 成交量特征
            features = self._add_volume_features(df, features)

            # 去除NaN值
            features = features.dropna()

            # 保存特征名称
            self.feature_names = features.columns.tolist()


            return features

        except Exception as e:
            logging.error(f"特征创建失败: {str(e)}")
            return None

    def _add_price_features(self, df, features):
        """价格相关特征"""
        # 价格差异
        features['price_range'] = df['high'] - df['low']
        features['price_range_ratio'] = features['price_range'] / df['close']

        # 价格变化
        features['price_change'] = df['close'].pct_change()
        features['price_change_5'] = df['close'].pct_change(5)
        features['price_change_10'] = df['close'].pct_change(10)
        features['price_change_20'] = df['close'].pct_change(20)

        # 对数收益率
        features['log_return'] = np.log(df['close'] / df['close'].shift(1))

        return features

    def _add_technical_indicators(self, df, features):
        """技术指标"""
        # 移动平均线
        for period in [5, 10, 20, 30, 60]:
            features[f'ma_{period}'] = talib.MA(df['close'], timeperiod=period)
            features[f'ma_diff_{period}'] = df['close'] - features[f'ma_{period}']

        # MACD
        features['macd'], features['macd_signal'], features['macd_hist'] = talib.MACD(
            df['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )

        # RSI
        for period in [6, 12, 24]:
            features[f'rsi_{period}'] = talib.RSI(df['close'], timeperiod=period)

        # KDJ
        features['slowk'], features['slowd'] = talib.STOCH(
            df['high'], df['low'], df['close'],
            fastk_period=9, slowk_period=3, slowk_matype=0,
            slowd_period=3, slowd_matype=0
        )

        # 布林带
        features['bb_upper'], features['bb_middle'], features['bb_lower'] = talib.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )

        return features

    def _add_volatility_indicators(self, df, features):
        """波动性指标"""
        # ATR - Average True Range
        features['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)

        # 历史波动率
        features['volatility_5'] = df['close'].rolling(window=5).std()
        features['volatility_10'] = df['close'].rolling(window=10).std()
        features['volatility_20'] = df['close'].rolling(window=20).std()

        return features

    def _add_trend_features(self, df, features):
        """趋势特征"""
        # ADX - Average Directional Index
        features['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)

        # AROON
        features['aroon_up'], features['aroon_down'] = talib.AROON(df['high'], df['low'], timeperiod=14)

        # CCI - Commodity Channel Index
        features['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)

        return features

    def _add_volume_features(self, df, features):
        """成交量特征"""
        # 成交量变化
        features['volume_change'] = df['vol'].pct_change()

        # 成交量移动平均
        for period in [5, 10, 20]:
            features[f'volume_ma_{period}'] = talib.MA(df['vol'], timeperiod=period)

        # 成交量相对强度
        features['volume_rsi'] = talib.RSI(df['vol'], timeperiod=14)

        # OBV - On Balance Volume
        features['obv'] = talib.OBV(df['close'], df['vol'])

        return features

    def normalize_features(self, features):
        """特征标准化"""
        normalized_features = pd.DataFrame(
            self.scaler.fit_transform(features),
            columns=features.columns,
            index=features.index
        )
        return normalized_features