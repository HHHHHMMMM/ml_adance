import logging

import chinadata.ca_data as ts
import pandas as pd

from lh_code_adv.config.config import Config
from lh_code_adv.utils.logger import log_method


class DataEngine:
    @log_method
    def __init__(self):
        self.pro = ts.pro_api(Config.TUSHARE_TOKEN)
        logging.info("初始化数据引擎...")
    @log_method
    def get_stock_list(self):
        """获取股票列表"""
        try:
            stocks = self.pro.stock_basic(
                exchange='',
                list_status='L',
                fields='ts_code,symbol,name,area,industry,list_date'
            )
            logging.info("获取数据成功...")
            if Config.EXCLUDE_ST:
                stocks = stocks[~stocks['name'].str.contains('ST')]

            return stocks
        except Exception as e:
            print(f"获取股票列表失败: {str(e)}")
            return None

    def get_daily_data(self, ts_code, start_date=None, end_date=None):
        """获取日线数据"""
        try:
            if start_date is None:
                start_date = Config.START_DATE
            if end_date is None:
                end_date = Config.END_DATE

            daily_data = self.pro.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields='ts_code,trade_date,open,high,low,close,vol,amount'
            )

            return daily_data
        except Exception as e:
            print(f"获取日线数据失败: {str(e)}")
            return None


    def index_daily(self, ts_code, start_date=None, end_date=None,fields='trade_date,close'):
        """获取日线数据"""
        try:
            if start_date is None:
                start_date = Config.START_DATE
            if end_date is None:
                end_date = Config.END_DATE

            index_daily_data = self.pro.index_daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields=fields
            )

            return index_daily_data
        except Exception as e:
            print(f"获取基准线数据失败: {str(e)}")
            return None



    def get_minute_data(self, ts_code, trade_date):
        """获取分钟级数据"""
        try:
            df = self.pro.stk_mins(
                ts_code=ts_code,
                trade_date=trade_date
            )
            return df
        except Exception as e:
            print(f"获取分钟数据失败: {str(e)}")
            return None

    def get_realtime_data(self, stock_list=None):
        """获取实时行情数据"""
        try:
            if stock_list is None:
                stock_list = self.get_stock_list()['ts_code'].tolist()

            # 使用tushare的实时行情接口
            df = self.pro.quotes(ts_code=','.join(stock_list))

            # 添加时间戳
            df['timestamp'] = pd.Timestamp.now()

            # 数据预处理
            df = self._preprocess_realtime_data(df)

            return df
        except Exception as e:
            logging.error(f"获取实时数据失败: {str(e)}")
            return None

    def _preprocess_realtime_data(self, df):
        """预处理实时数据"""
        if df is None or df.empty:
            return None

        # 重命名列以匹配系统其他部分的命名约定
        column_mapping = {
            'current': 'close',
            'vol': 'volume',
            'amount': 'trade_amount'
        }
        df = df.rename(columns=column_mapping)

        # 确保数据类型正确
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'trade_amount']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df