# data/data_processor.py

import pandas as pd
import numpy as np


class DataProcessor:
    @staticmethod
    def process_daily_data(df):
        """处理日线数据"""
        if df is None or df.empty:
            return None

        df = df.copy()

        # 按日期排序
        df = df.sort_values('trade_date')

        # 处理缺失值
        df = df.fillna(method='ffill')

        # 添加基础计算列
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        return df

    @staticmethod
    def process_minute_data(df):
        """处理分钟数据"""
        if df is None or df.empty:
            return None

        # 处理分钟数据的特定逻辑
        df = df.sort_values('trade_time')
        df = df.fillna(method='ffill')

        return df