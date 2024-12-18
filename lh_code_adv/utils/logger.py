import logging
import functools
import time
import os
from datetime import datetime


def setup_logger(name='quant_trading'):
    """设置日志配置"""
    # 创建logs目录
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 生成日志文件名
    log_filename = os.path.join(log_dir, f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    # 设置日志格式
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s'
    )

    # 文件处理器
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setFormatter(formatter)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # 获取日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 确保处理器不会重复添加
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    # 阻止日志传播到父记录器
    logger.propagate = False

    return logger


def log_method(func):
    """方法执行日志装饰器"""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        logger = logging.getLogger('quant_trading')
        func_name = func.__name__
        class_name = self.__class__.__name__

        logger.info(f"[{class_name}] 开始执行: {func_name}")
        start_time = time.time()

        try:
            result = func(self, *args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"[{class_name}] 成功完成: {func_name}, 耗时: {duration:.2f}秒")
            return result
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            logger.error(f"[{class_name}] 执行失败: {func_name}, 耗时: {duration:.2f}秒, 错误: {str(e)}")
            raise

    return wrapper


class ProgressLogger:
    """进度日志记录器"""

    def __init__(self, total, desc="Progress", log_interval=1):
        self.logger = logging.getLogger('quant_trading')
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = time.time()
        self.log_interval = log_interval
        self.last_log_time = self.start_time

    def update(self, n=1):
        self.current += n
        current_time = time.time()

        if (current_time - self.last_log_time) >= self.log_interval:
            progress = (self.current / self.total) * 100
            elapsed = current_time - self.start_time
            speed = self.current / elapsed if elapsed > 0 else 0
            eta = (self.total - self.current) / speed if speed > 0 else 0

            self.logger.info(
                f"{self.desc}: {self.current}/{self.total} "
                f"({progress:.1f}%) "
                f"速度: {speed:.1f}项/秒 "
                f"预计剩余时间: {eta:.1f}秒"
            )

            self.last_log_time = current_time

    def close(self):
        end_time = time.time()
        total_time = end_time - self.start_time
        self.logger.info(
            f"{self.desc}完成: 总计 {self.current} 项, "
            f"总耗时: {total_time:.2f}秒"
        )