import pandas as pd
import sqlite3
from datetime import datetime
import logging


class DataManager:
    def __init__(self, db_path='data/market_data.db'):
        self.db_path = db_path
        self.conn = None
        self.initialize_database()

    def initialize_database(self):
        """初始化数据库"""
        try:
            self.conn = sqlite3.connect(self.db_path)

            # 创建必要的表
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS daily_data (
                    ts_code TEXT,
                    trade_date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    amount REAL,
                    PRIMARY KEY (ts_code, trade_date)
                )
            ''')

            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS features (
                    ts_code TEXT,
                    trade_date TEXT,
                    feature_name TEXT,
                    value REAL,
                    PRIMARY KEY (ts_code, trade_date, feature_name)
                )
            ''')

            self.conn.commit()
        except Exception as e:
            logging.error(f"数据库初始化失败: {str(e)}")
            raise

    def update_realtime_data(self, ts_code, data):
        """更新实时数据"""
        try:
            # 创建实时数据表(如果不存在)
            self.conn.execute('''
                   CREATE TABLE IF NOT EXISTS realtime_data (
                       ts_code TEXT,
                       timestamp TEXT,
                       price REAL,
                       volume REAL,
                       bid_price REAL,
                       ask_price REAL,
                       PRIMARY KEY (ts_code, timestamp)
                   )
               ''')

            # 插入或更新实时数据
            sql = '''
                   INSERT OR REPLACE INTO realtime_data 
                   (ts_code, timestamp, price, volume, bid_price, ask_price)
                   VALUES (?, ?, ?, ?, ?, ?)
               '''
            self.conn.execute(sql, (
                ts_code,
                data['timestamp'],
                data['price'],
                data['volume'],
                data['bid_price'],
                data['ask_price']
            ))
            self.conn.commit()

        except Exception as e:
            logging.error(f"更新实时数据失败: {str(e)}")
            raise


    def save_daily_data(self, data):
        """保存日线数据"""
        try:
            data.to_sql('daily_data', self.conn, if_exists='append', index=False)
        except Exception as e:
            logging.error(f"保存日线数据失败: {str(e)}")
            raise

    def clean_historical_data(self):
        """清理历史数据"""
        try:
            # 删除超过N天的实时数据
            retention_days = 30
            sql = f'''
                DELETE FROM realtime_data 
                WHERE timestamp < date('now', '-{retention_days} days')
            '''
            self.conn.execute(sql)

            # 优化数据库
            self.conn.execute('VACUUM')
            self.conn.commit()

        except Exception as e:
            logging.error(f"清理历史数据失败: {str(e)}")
            raise

    def save_features(self, features, ts_code, trade_date):
        """保存特征数据"""
        try:
            feature_df = pd.DataFrame({
                'ts_code': ts_code,
                'trade_date': trade_date,
                'feature_name': features.keys(),
                'value': features.values()
            })
            feature_df.to_sql('features', self.conn, if_exists='append', index=False)
        except Exception as e:
            logging.error(f"保存特征数据失败: {str(e)}")
            raise

    def get_data_for_training(self, start_date, end_date):
        """获取训练数据"""
        query = f"""
            SELECT * FROM daily_data 
            WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY trade_date
        """
        return pd.read_sql(query, self.conn)

    def get_latest_features(self, ts_code):
        """获取最新特征"""
        query = f"""
            SELECT feature_name, value 
            FROM features 
            WHERE ts_code = '{ts_code}'
            AND trade_date = (
                SELECT MAX(trade_date) 
                FROM features 
                WHERE ts_code = '{ts_code}'
            )
        """
        return pd.read_sql(query, self.conn)


def export_data(self, start_date, end_date, format='csv'):
    """导出数据"""
    try:
        # 获取数据
        sql = f'''
                SELECT d.*, f.feature_name, f.value
                FROM daily_data d
                LEFT JOIN features f ON d.ts_code = f.ts_code 
                    AND d.trade_date = f.trade_date
                WHERE d.trade_date BETWEEN ? AND ?
            '''
        df = pd.read_sql(sql, self.conn, params=[start_date, end_date])

        # 数据透视
        df_pivot = df.pivot_table(
            index=['ts_code', 'trade_date'],
            columns='feature_name',
            values=['open', 'high', 'low', 'close', 'volume', 'amount', 'value']
        ).reset_index()

        # 导出数据
        if format == 'csv':
            df_pivot.to_csv(f'export_data_{start_date}_{end_date}.csv', index=False)
        elif format == 'excel':
            df_pivot.to_excel(f'export_data_{start_date}_{end_date}.xlsx', index=False)

        return df_pivot

    except Exception as e:
        logging.error(f"导出数据失败: {str(e)}")
        raise