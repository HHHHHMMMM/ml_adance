# monitoring/trading_monitor.py

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from lh_code_adv.config.config import Config


class TradingMonitor:
    def __init__(self):
        self.logger = logging.getLogger('trading_monitor')
        self.performance_metrics = []
        self.last_alert_time = {}

    def check_market_status(self):
        """检查市场状态"""
        current_time = datetime.now().time()
        market_open = datetime.strptime(Config.MARKET_OPEN_TIME, "%H:%M:%S").time()
        market_close = datetime.strptime(Config.MARKET_CLOSE_TIME, "%H:%M:%S").time()

        # 检查是否为交易时间
        is_trading_time = market_open <= current_time <= market_close

        # 检查是否为工作日
        is_workday = datetime.now().weekday() < 5

        return is_trading_time and is_workday

    def monitor_performance(self, portfolio_value, positions):
        """监控交易表现"""
        current_time = datetime.now()

        # 计算性能指标
        metrics = {
            'timestamp': current_time,
            'portfolio_value': portfolio_value,
            'position_count': len(positions),
            'daily_return': self._calculate_daily_return(portfolio_value)
        }

        self.performance_metrics.append(metrics)

        # 检查是否需要发送警报
        self._check_alerts(metrics)

        # 如果开启性能分析
        if Config.ENABLE_PROFILING:
            self._log_performance_stats()

    def _calculate_daily_return(self, current_value):
        """计算日收益率"""
        if not self.performance_metrics:
            return 0

        prev_day = (datetime.now() - timedelta(days=1)).date()
        prev_value = next(
            (m['portfolio_value'] for m in reversed(self.performance_metrics)
             if m['timestamp'].date() == prev_day),
            current_value
        )

        return (current_value - prev_value) / prev_value if prev_value else 0

    def _check_alerts(self, metrics):
        """检查是否需要发送警报"""
        alerts = []

        # 检查大幅波动
        if abs(metrics['daily_return']) > Config.MAX_DAILY_RETURN:
            alerts.append(f"日收益率波动过大: {metrics['daily_return']:.2%}")

        # 检查持仓集中度
        if metrics['position_count'] < 3:
            alerts.append(f"持仓过于集中，当前持仓数: {metrics['position_count']}")

        # 发送警报
        if alerts and Config.ENABLE_EMAIL_ALERT:
            self._send_alert_email(alerts)

    def _send_alert_email(self, alerts):
        """发送警报邮件"""
        try:
            # 控制邮件发送频率
            current_time = datetime.now()
            if 'alert_email' in self.last_alert_time:
                if (current_time - self.last_alert_time['alert_email']).total_seconds() < 3600:
                    return

            msg = MIMEMultipart()
            msg['Subject'] = f"交易系统警报 - {current_time.strftime('%Y-%m-%d %H:%M:%S')}"
            msg['From'] = Config.SMTP_USER
            msg['To'] = ', '.join(Config.ALERT_RECIPIENTS)

            body = "\n".join([f"- {alert}" for alert in alerts])
            msg.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP(Config.SMTP_SERVER, Config.SMTP_PORT) as server:
                server.starttls()
                server.login(Config.SMTP_USER, Config.SMTP_PASSWORD)
                server.send_message(msg)

            self.last_alert_time['alert_email'] = current_time
            self.logger.info("警报邮件已发送")

        except Exception as e:
            self.logger.error(f"发送警报邮件失败: {str(e)}")

    def _log_performance_stats(self):
        """记录性能统计"""
        if not self.performance_metrics:
            return

        # 计算统计数据
        recent_metrics = pd.DataFrame(self.performance_metrics[-100:])
        stats = {
            'avg_portfolio_value': recent_metrics['portfolio_value'].mean(),
            'daily_return_std': recent_metrics['daily_return'].std(),
            'position_count_avg': recent_metrics['position_count'].mean()
        }

        # 记录到日志
        self.logger.info(f"性能统计: {stats}")

    def cleanup_old_metrics(self):
        """清理旧的性能指标数据"""
        if len(self.performance_metrics) > 1000:
            self.performance_metrics = self.performance_metrics[-1000:]