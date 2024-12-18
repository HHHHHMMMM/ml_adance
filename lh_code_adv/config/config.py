# config/config.py

import os
from datetime import datetime, timedelta


class Config:
    # 基础配置
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
    MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

    # Tushare配置
    TUSHARE_TOKEN = "n9e84ed87f29cf43fdac84cdbb14d306777"

    # 数据配置
    START_DATE = (datetime.now() - timedelta(days=365 * 5)).strftime('%Y%m%d')  # 5年数据
    END_DATE = datetime.now().strftime('%Y%m%d')

    # 股票池配置
    EXCLUDE_ST = True
    MIN_PRICE = 5.0

    # 交易配置
    TRADE_TYPE = "daily"  # daily/intraday
    COMMISSION_RATE = 0.0003  # 手续费率
    SLIPPAGE_RATE = 0.0001  # 滑点率
    POSITION_LIMIT = 0.8  # 最大仓位限制
    MIN_HOLDINGS = 1  # 最小持仓数
    MAX_HOLDINGS = 10  # 最大持仓数

    # 模型配置
    SEQUENCE_LENGTH = 30
    BATCH_SIZE = 64
    EPOCHS = 100
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    DROPOUT = 0.2

    # 风险控制
    MAX_DRAWDOWN = 0.15
    STOP_LOSS = 0.05

    # 回测配置
    INITIAL_CAPITAL = 1000000
    BENCHMARK = '000300.SH'  # 沪深300作为基准

    # 策略参数
    BUY_THRESHOLD = 0.7
    SELL_THRESHOLD = 0.3

    # 风险控制参数
    MAX_PORTFOLIO_VAR = 0.2
    MAX_DRAWDOWN = 0.15

    # 实时交易参数
    REFRESH_INTERVAL = 5  # 秒


    # 数据配置
    TRAINING_START_DATE = "20180101"  # 训练数据开始日期
    TRAINING_END_DATE = "20231231"  # 训练数据结束日期

    # 模型配置
    TRAIN_TEST_SPLIT = 0.8  # 训练集比例
    RANDOM_SEED = 42  # 随机种子

    # 实时交易配置
    REFRESH_INTERVAL = 5  # 实时数据刷新间隔(秒)
    BUY_THRESHOLD = 0.7  # 买入阈值
    SELL_THRESHOLD = 0.3  # 卖出阈值

    # 风控配置
    MAX_POSITION_SIZE = 0.1  # 单个股票最大仓位
    MAX_DRAWDOWN = 0.15  # 最大回撤限制
    STOP_LOSS = 0.05  # 止损比例

    # 预警配置
    ALERT_METHODS = ['log', 'email']  # 预警方式
    EMAIL_SETTINGS = {
        'smtp_server': 'smtp.example.com',
        'smtp_port': 587,
        'sender_email': 'your_email@example.com',
        'password': 'your_password'
    }

    INITIAL_CAPITAL = 1000000  # 初始资金

    # 数据配置
    START_DATE = "20200101"
    END_DATE = "20231231"
    EXCLUDE_ST = True  # 是否排除ST股票

    # 实时交易配置
    REFRESH_INTERVAL = 60  # 实时数据刷新间隔（秒）
    RETRY_INTERVAL = 10  # 重试间隔（秒）
    MARKET_OPEN_TIME = "09:30:00"
    MARKET_CLOSE_TIME = "15:00:00"

    # 交易参数
    MIN_TRADE_AMOUNT = 100  # 最小交易数量
    COMMISSION_RATE = 0.0003  # 手续费率
    SLIPPAGE_RATE = 0.0002  # 滑点率
    POSITION_SIZE = 0.1  # 单个持仓占总资产的最大比例

    # 风险控制参数
    MAX_DRAWDOWN = 0.1  # 最大回撤限制
    STOP_LOSS = 0.05  # 止损比例
    POSITION_LIMIT = 0.2  # 单个持仓限制
    MAX_CONCENTRATION = 0.3  # 最大持仓集中度

    # 交易信号阈值
    BUY_THRESHOLD = 0.7  # 买入信号阈值
    SELL_THRESHOLD = 0.7  # 卖出信号阈值

    # 数据库配置
    DB_HOST = "localhost"
    DB_PORT = 3306
    DB_USER = "root"
    DB_PASSWORD = "password"
    DB_NAME = "quant_trading"

    # 日志配置
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = "trading_system.log"

    # 缓存配置
    CACHE_DIR = "./cache"
    MAX_CACHE_SIZE = 1000  # 最大缓存条目数

    # 性能监控
    ENABLE_PROFILING = False
    PROFILING_INTERVAL = 300  # 性能统计间隔（秒）

    # 邮件通知配置
    ENABLE_EMAIL_ALERT = False
    SMTP_SERVER = "smtp.example.com"
    SMTP_PORT = 587
    SMTP_USER = "your_email@example.com"
    SMTP_PASSWORD = "your_password"
    ALERT_RECIPIENTS = ["alert@example.com"]

    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        for directory in [cls.DATA_DIR, cls.LOG_DIR, cls.MODEL_DIR]:
            if not os.path.exists(directory):
                os.makedirs(directory)


