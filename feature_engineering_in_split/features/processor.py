from typing import Dict

import pandas as pd
import logging



class DataTypeProcessor:
    """数据类型处理器"""

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.dtype_mappings = {
            'int': ['int8', 'int16', 'int32', 'int64'],
            'float': ['float16', 'float32', 'float64'],
            'category': ['object', 'category'],
            'datetime': ['datetime64[ns]', 'datetime64']
        }

    def infer_and_convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """推断并转换数据类型"""
        df = df.copy()
        for column in df.columns:
            try:
                if df[column].dtype in self.dtype_mappings['int'] + self.dtype_mappings['float']:
                    df[column] = self._optimize_numeric_type(df[column])
                elif self._is_datetime(df[column]):
                    df[column] = pd.to_datetime(df[column], errors='coerce')
                elif self._should_be_category(df[column]):
                    if df[column].isna().sum() / len(df[column]) < 0.05:
                        df[column] = df[column].astype('category')
            except Exception as e:
                self.logger.warning(f"列 {column} 类型转换失败: {str(e)}")
                continue
        return df

    def check_data_types(self, df: pd.DataFrame) -> Dict:
        """检查数据类型并返回报告"""
        report = {
            'invalid_types': [],
            'type_distribution': {},
            'memory_usage': {},
            'recommendations': []
        }

        for column in df.columns:
            curr_type = str(df[column].dtype)
            memory_usage = df[column].memory_usage(deep=True) / 1024 / 1024  # MB

            report['type_distribution'][column] = curr_type
            report['memory_usage'][column] = f"{memory_usage:.2f} MB"

            # 检查无效值
            if df[column].isna().any():
                report['invalid_types'].append(f"{column}: 包含缺失值")

            # 提供优化建议
            if curr_type in self.dtype_mappings['float']:
                if self._can_downcast_to_int(df[column]):
                    report['recommendations'].append(
                        f"{column}: 可以转换为整数类型以节省内存"
                    )
            elif curr_type in self.dtype_mappings['category']:
                if not self._should_be_category(df[column]):
                    report['recommendations'].append(
                        f"{column}: 可能不适合作为分类类型"
                    )

        return report
