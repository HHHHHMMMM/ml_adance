# monitor/monitor_engine.py

import pandas as pd
from datetime import datetime
import logging
import json
from ..config.config import Config


class MonitorEngine:
    def __init__(self):
        self.performance_metrics = []
        self.risk_metrics = []
        self.alerts = []

    def update_performance(self, metrics):
        """
        更新性能指标

        Parameters:
        metrics (dict): 性能指标
        """
        metrics['timestamp'] = datetime.now()
        self.performance_metrics.append(metrics)

        # 检查性能预警
        self._check_performance_alerts(metrics)

    def update_risk_metrics(self, metrics):
        """
        更新风险指标

        Parameters:
        metrics (dict): 风险指标
        """
        metrics['timestamp'] = datetime.now()
        self.risk_metrics.append(metrics)

        # 检查风险预警
        self._check_risk_alerts(metrics)

    def add_alert(self, alert_type, message, severity='INFO'):
        """添加预警信息"""
        alert = {
            'type': alert_type,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now()
        }

        self.alerts.append(alert)

        # 记录日志
        if severity == 'ERROR':
            logging.error(message)
        elif severity == 'WARNING':
            logging.warning(message)
        else:
            logging.info(message)

    def _check_performance_alerts(self, metrics):
        """检查性能预警"""
        # 检查回撤
        if metrics.get('drawdown', 0) > Config.MAX_DRAWDOWN:
            self.add_alert(
                'DRAWDOWN',
                f"Maximum drawdown exceeded: {metrics['drawdown']:.2%}",
                'WARNING'
            )

        # 检查收益率
        if metrics.get('daily_return', 0) < -0.05:  # 单日跌幅超过5%
            self.add_alert(
                'RETURN',
                f"Significant daily loss: {metrics['daily_return']:.2%}",
                'WARNING'
            )

    def _check_risk_alerts(self, metrics):
        """检查风险预警"""
        # 检查持仓集中度
        if metrics.get('concentration_risk', 0) > 0.5:  # 单个持仓超过50%
            self.add_alert(
                'CONCENTRATION',
                f"High position concentration: {metrics['concentration_risk']:.2%}",
                'WARNING'
            )

    def get_summary(self):
        """获取监控摘要"""
        return {
            'latest_performance': self.performance_metrics[-1] if self.performance_metrics else None,
            'latest_risk': self.risk_metrics[-1] if self.risk_metrics else None,
            'recent_alerts': self.alerts[-10:]  # 最近10条预警
        }

    def export_report(self, filepath):
        """导出监控报告"""
        report = {
            'performance_metrics': self.performance_metrics,
            'risk_metrics': self.risk_metrics,
            'alerts': self.alerts
        }

        with open(filepath, 'w') as f:
            json.dump(report, f, default=str, indent=4)