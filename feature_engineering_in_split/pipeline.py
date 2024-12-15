import os
import logging
import time
from typing import Dict, Tuple, List, Optional
import pandas as pd
import numpy as np
import torch
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pickle

from utils.monitoring import MemoryMonitor
from features.processor import DataTypeProcessor
from models.trainer import ModelTrainer
from features.extractor import DeepFeatureExtractor


class MLPipeline:
    """机器学习流水线"""

    def __init__(self, task_type: str = 'classification',
                 random_state: int = 42,
                 device: str = None):
        self.task_type = task_type
        self.random_state = random_state
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = self._init_logger()
        self.data_processor = DataTypeProcessor(logger=self.logger)
        self.memory_monitor = MemoryMonitor(logger=self.logger, threshold_mb=1000)
        self.feature_extractors = {}
        self.models = {}
        self.best_model = None

    def _init_logger(self) -> logging.Logger:
        """初始化日志记录器"""
        logger = logging.getLogger('MLPipeline')
        for handler in logger.handlers[:]:  # 复制处理器列表进行迭代
            logger.removeHandler(handler)
            # 添加新的处理器
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _load_and_preprocess_data(self, data_dir: str, target_col: str, config: Dict, stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame,pd.Series, pd.Series]:

        """数据加载和预处理"""
        self.logger.info("开始数据加载和预处理...")

        # 查找训练文件
        train_files = [f for f in os.listdir(data_dir) if f.startswith('train')]
        if not train_files:
            raise FileNotFoundError(f"未找到训练数据文件。目录 {data_dir}中内容: {os.listdir(data_dir)}")

        # 读取数据
        train_file = os.path.join(data_dir, train_files[0])
        self.logger.info(f"正在读取训练文件: {train_file}")
        file_ext = os.path.splitext(train_file)[1]
        try:
            if file_ext == '.csv':
                df = pd.read_csv(train_file)
            elif file_ext in ['.xls', '.xlsx']:
                df = pd.read_excel(train_file)
            elif file_ext == '.parquet':
                df = pd.read_parquet(train_file)
            else:
                raise ValueError(f"不支持的文件格式: {file_ext}")

            # 检查数据完整性
            if df.empty:
                raise ValueError("加载的数据集为空")
            self.logger.info(f"数据加载完成，形状: {df.shape}")

            # 检查目标列
            if target_col not in df.columns:
                raise ValueError(f"目标列 '{target_col}' 不在数据中")

            # 分离特征和目标
            X = df.drop(columns=[target_col])
            y = df[target_col]
            X_train = X
            y_train = y

            # 训练集验证集分割
            val_files = [f for f in os.listdir(data_dir) if f.startswith('vali')]
            if val_files:
                val_file = val_files[0]
                val_df = pd.read_csv(os.path.join(data_dir, val_file))
                if target_col not in val_df.columns:
                    raise ValueError(f"验证集中未找到目标列 '{target_col}'")
                X_val = val_df.drop(columns=[target_col])
                y_val = val_df[target_col]
            else:
                # 确保分层采样的正确实现
                stratify_param = y if stratify and self.task_type == 'classification' else None
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y,
                    test_size=config.get('test_size', 0.2),
                    random_state=self.random_state,
                    stratify=stratify_param
                )

            # 特征一致性检查
            if not set(X_train.columns) == set(X_val.columns):
                raise ValueError("训练集和验证集的特征不一致")

            return X_train, X_val, y_train, y_val

        except Exception as e:
            self.logger.error(f"数据加载失败: {str(e)}")
            raise


    def readFiles(self,data_path:str, target_col: str)->Tuple[pd.DataFrame,pd.Series]:
        train_data=[f for f in os.listdir(data_path) if f.startswith('train')]
        train_file = os.path.join(data_path, train_data[0])
        self.logger.info(f"正在读取训练文件: {train_file}")
        file_ext = os.path.splitext(train_file)[1]
        try:
            if file_ext == '.csv':
                df = pd.read_csv(train_file)
            elif file_ext in ['.xls', '.xlsx']:
                df = pd.read_excel(train_file)
            elif file_ext == '.parquet':
                df = pd.read_parquet(train_file)
            else:
                raise ValueError(f"不支持的文件格式: {file_ext}")

            # 检查数据完整性
            if df.empty:
                raise ValueError("加载的数据集为空")
            self.logger.info(f"数据加载完成，形状: {df.shape}")

            # 检查目标列
            if target_col not in df.columns:
                raise ValueError(f"目标列 '{target_col}' 不在数据中")
        except Exception as e:
            self.logger.error(f"数据加载失败: {str(e)}")
            raise


    def preprocess_with_monitoring(self,
                                   df: pd.DataFrame,
                                   target_col: str) -> Tuple[pd.DataFrame, pd.Series, Dict, Dict]:
        """带监控的数据预处理"""
        self.memory_monitor.start_monitoring()
        self.logger.info("开始数据预处理...")

        # 记录初始内存使用
        initial_memory = self.memory_monitor.check_memory(log_usage=True)

        # 优化数据类型
        df_optimized = self.data_processor.infer_and_convert_types(df)

        # 获取数据类型报告
        type_report = self.data_processor.check_data_types(df_optimized)
        self.logger.info(f"数据类型优化报告: {type_report}")

        # 分离特征和目标
        X = df_optimized.drop(columns=[target_col])
        y = df_optimized[target_col]

        # 记录最终内存使用
        memory_profile = self.memory_monitor.get_memory_profile()

        return X, y, memory_profile, type_report

    def analyze_data(self, x_train: pd.DataFrame,
                             x_val: pd.DataFrame, target_col: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series, Dict, Dict,Dict]:
        """
        数据分析函数

        Args:
            df: 待分析的数据框
            target_col: 目标变量列名
            :param X_val:
        """
        train_data = x_train.copy()
        val_data = x_val.copy()
        complete_data = pd.concat([train_data, val_data])
        memory_stats = self.memory_monitor.check_memory()
        self.logger.info(f"数据合并后内存使用: {memory_stats['memory_used_mb']:.2f}MB")

        self.logger.info(f"开始数据分析，数据形状: {complete_data.shape}")
        self.logger.info(f"数据列名: {complete_data.columns.tolist()}")

        if target_col and target_col not in complete_data.columns:
            raise ValueError(f"目标列 '{target_col}' 不在数据集中。可用的列有: {complete_data.columns.tolist()}")

        analysis_results = {}

        # 基本统计信息
        analysis_results['basic_stats'] = complete_data.describe()

        # 区分特征类型
        numeric_cols = complete_data.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = complete_data.select_dtypes(include=['object', 'category']).columns

        analysis_results['feature_types'] = {
            'numeric': list(numeric_cols),
            'categorical': list(categorical_cols)
        }

        # 空值分析
        null_analysis = complete_data.isnull().mean() * 100
        analysis_results['null_percentage'] = null_analysis

        # 如果是分类任务，分析目标变量分布
        if target_col and self.task_type == 'classification':
            class_distribution = complete_data[target_col].value_counts(normalize=True)
            analysis_results['class_distribution'] = class_distribution

            # 判断是否存在类别不平衡
            class_imbalance = max(class_distribution) / min(class_distribution)
            analysis_results['class_imbalance_ratio'] = class_imbalance

            if class_imbalance > 3:
                self.logger.warning("检测到显著的类别不平衡问题")
                analysis_results['rebalancing_methods'] = [
                    {
                        'method': 'SMOTE',
                        'description': '合成少数类过采样技术',
                        'pros': '能生成新的少数类样本',
                        'cons': '可能导致过拟合'
                    },
                    {
                        'method': 'RandomUnderSampler',
                        'description': '随机欠采样',
                        'pros': '简单快速',
                        'cons': '可能丢失重要信息'
                    },
                    {
                        'method': 'class_weight',
                        'description': '使用类别权重',
                        'pros': '不改变数据分布',
                        'cons': '可能需要精细调整'
                    }
                ]

            X, y, memory_profile, type_report = self.preprocess_with_monitoring(df=complete_data, target_col=target_col)


        return X, y, memory_profile, type_report,analysis_results


    def _feature_engineering(self, X_train: pd.DataFrame,
                             X_val: pd.DataFrame,
                             config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """特征工程"""
        self.logger.info("开始特征工程...")
        X_train_processed = X_train.copy()
        X_val_processed = X_val.copy()

        # 1. 处理分类特征
        categorical_features = X_train.select_dtypes(include=['object', 'category']).columns
        label_encoders = {}

        for col in categorical_features:
            label_encoders[col] = LabelEncoder()
            X_train_processed[col] = label_encoders[col].fit_transform(X_train_processed[col])
            X_val_processed[col] = label_encoders[col].transform(X_val_processed[col])

        # 2. 特征缩放
        if config.get('scale_features', True):
            numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
            scaler = StandardScaler()
            X_train_processed[numeric_features] = scaler.fit_transform(X_train_processed[numeric_features])
            X_val_processed[numeric_features] = scaler.transform(X_val_processed[numeric_features])

        # 3. 深度特征提取（如果配置启用）
        if config.get('use_deep_features', False):
            try:
                deep_extractor = DeepFeatureExtractor(
                    input_dim=X_train_processed.shape[1],
                    hidden_dims=config.get('deep_feature_config', {}).get('hidden_dims', [256, 128, 64]),
                    dropout_rate=config.get('deep_feature_config', {}).get('dropout_rate', 0.3)
                ).to(self.device)

                # 转换为PyTorch张量
                X_train_tensor = torch.FloatTensor(X_train_processed.values).to(self.device)
                X_val_tensor = torch.FloatTensor(X_val_processed.values).to(self.device)

                # 提取深度特征
                with torch.no_grad():
                    deep_features_train = deep_extractor.encoder(X_train_tensor).cpu().numpy()
                    deep_features_val = deep_extractor.encoder(X_val_tensor).cpu().numpy()

                # 添加深度特征
                for i in range(deep_features_train.shape[1]):
                    X_train_processed[f'deep_feature_{i}'] = deep_features_train[:, i]
                    X_val_processed[f'deep_feature_{i}'] = deep_features_val[:, i]

                self.feature_extractors['deep'] = deep_extractor
                self.logger.info(f"深度特征提取完成，新增特征数: {deep_features_train.shape[1]}")

            except Exception as e:
                self.logger.error(f"深度特征提取失败: {str(e)}")

        return X_train_processed, X_val_processed

    def _train_and_evaluate_models(self, X_train: pd.DataFrame,
                                   X_val: pd.DataFrame,
                                   y_train: pd.Series,
                                   y_val: pd.Series,
                                   config: Dict) -> Dict:
        """模型训练和评估"""
        self.logger.info("开始模型训练和评估...")
        results = {}

        # 获取要训练的模型列表
        models_to_train = config.get('models', ['xgb', 'catboost', 'lgb'])

        for model_name in models_to_train:
            self.logger.info(f"\n开始训练模型: {model_name}")

            try:
                # 获取模型配置
                model_config = config.get('param_grids', {}).get(model_name, {})

                # 初始化模型训练器
                trainer = ModelTrainer(
                    model_type=model_name,
                    config=model_config,
                    logger=self.logger
                )

                # 训练和评估
                model_results = trainer.train(
                    X_train, y_train,
                    X_val, y_val
                )

                # 保存结果和模型
                results[model_name] = model_results
                self.models[model_name] = trainer.model

                # 更新最佳模型
                if self.best_model is None or \
                        model_results.get('val_accuracy', 0) > \
                        results.get(self.best_model, {}).get('val_accuracy', 0):
                    self.best_model = model_name

            except Exception as e:
                self.logger.error(f"{model_name} 训练失败: {str(e)}")
                results[model_name] = {'error': str(e)}

        return results

    def _save_results(self, results: Dict, output_dir: str):
        """保存结果"""
        os.makedirs(output_dir, exist_ok=True)

        # 保存结果报告
        report_path = os.path.join(output_dir, 'results_report.pkl')
        with open(report_path, 'wb') as f:
            pickle.dump(results, f)

        # 保存最佳模型
        if self.best_model:
            model_path = os.path.join(output_dir, f'best_model_{self.best_model}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(self.models[self.best_model], f)

        # 保存特征提取器
        if self.feature_extractors:
            extractor_path = os.path.join(output_dir, 'feature_extractors.pkl')
            with open(extractor_path, 'wb') as f:
                pickle.dump(self.feature_extractors, f)

        self.logger.info(f"结果已保存至: {output_dir}")

    def run_pipeline(self, data_dir: str, target_col: str,
                     output_dir: str, config: Dict) -> Dict:
        """运行完整的机器学习流水线"""
        try:
            if not hasattr(self, 'memory_monitor'):
                self.memory_monitor = MemoryMonitor(threshold_mb=1000)
            # 开始内存监控
            self.memory_monitor.start_monitoring()
            start_time = time.time()

            # 1. 数据加载和预处理
            X_train, X_val, y_train, y_val = self._load_and_preprocess_data(
                data_dir, target_col, config,stratify=config.get('stratify', True)
            )


            # 2. 预数据分析
            analysis_results = self.analyze_data(
                x_train=X_train,
                x_val=X_val,
                target_col=target_col
            )


            # 2. 特征工程
            X_train_processed, X_val_processed = self._feature_engineering(
                X_train, X_val, config
            )

            # 3. 模型训练和评估
            model_results = self._train_and_evaluate_models(
                X_train_processed, X_val_processed,
                y_train, y_val, config
            )

            # 4. 整理结果
            pipeline_results = {
                'model_results': model_results,
                'best_model': self.best_model,
                'memory_usage': self.memory_monitor.get_memory_profile(),
                'execution_time': time.time() - start_time
            }

            # 5. 保存结果
            self._save_results(pipeline_results, output_dir)

            return pipeline_results

        except Exception as e:
            self.logger.error(f"Pipeline执行失败: {str(e)}")
            raise


def main():
    # 示例配置
    config = {
        'data': {
            'test_size': 0.2,
            'stratify': True
        },
        'models': ['xgb', 'catboost', 'lgb'],
        'use_deep_features': True,
        'scale_features': True,
        'deep_feature_config': {
            'hidden_dims': [256, 128, 64],
            'dropout_rate': 0.3
        },
        'output_dir': 'output'
    }

    # 初始化并运行pipeline
    pipeline = MLPipeline(
        task_type='classification',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    try:
        results = pipeline.run_pipeline(
            data_dir='data',
            target_col='target',
            output_dir='output',
            config=config
        )

        print("\n=== Pipeline执行完成 ===")
        print(f"最佳模型: {results['best_model']}")
        print(f"执行时间: {results['execution_time']:.2f}秒")

    except Exception as e:
        print(f"Pipeline执行失败: {str(e)}")


if __name__ == "__main__":
    main()
