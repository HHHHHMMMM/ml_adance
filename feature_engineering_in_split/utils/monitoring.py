import time

from typing import Optional, Union, List, Dict, Tuple

import logging



class MemoryMonitor:
    """内存使用监控器"""
    def __init__(self, threshold_mb: float = 1000, logger=None):
        self.threshold_mb = threshold_mb
        self.logger = logger or logging.getLogger(__name__)
        self.memory_usage = []
        self.start_time = None
        self.warnings_count = 0

    def start_monitoring(self):
        self.start_time = time.time()
        self.memory_usage = []
        self.warnings_count = 0

    def check_memory(self, log_usage: bool = True) -> Dict:
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            current_memory = memory_info.rss / 1024 / 1024

            memory_stats = {
                'timestamp': time.time(),
                'memory_used_mb': current_memory,
                'memory_percent': process.memory_percent()
            }

            self.memory_usage.append(memory_stats)
            return memory_stats
        except Exception as e:
            self.logger.error(f"内存监控失败: {str(e)}")
            return {}