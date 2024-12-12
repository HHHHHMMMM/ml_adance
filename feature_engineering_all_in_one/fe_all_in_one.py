import os
import time


import numpy as np
import pandas as pd
from typing import Optional, Union, List, Dict, Tuple
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, mean_squared_error, \
    mean_absolute_error
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pickle
import logging
from datetime import datetime
import warnings
from dataclasses import dataclass, field


class DeepFeatureExtractor(nn.Module):
    """深度学习特征提取器，包含编码器和解码器"""

    def __init__(self, input_dim: int, hidden_dims: List[int], dropout_rate: float = 0.3):
        super().__init__()

        # 编码器
        encoder_layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            current_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # 解码器
        decoder_layers = []
        hidden_dims.reverse()  # 反转隐藏层维度，构造解码器
        for hidden_dim in hidden_dims:
            decoder_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ])
            current_dim = hidden_dim
        decoder_layers.append(nn.Linear(current_dim, input_dim))  # 恢复到原始维度
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class DeepFeatureExtraction:
    """深度特征提取管理类"""

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [256, 128, 64],
                 batch_size: int = 32,
                 epochs: int = 50,
                 learning_rate: float = 1e-3,
                 dropout_rate: float = 0.3,
                 device: str = None):

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"正在使用{self.device}进行深度学习特征提取")
        self.model = DeepFeatureExtractor(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=5, verbose=True
        )

    def _prepare_data(self, X: np.ndarray) -> DataLoader:
        """准备数据加载器"""
        tensor_x = torch.FloatTensor(X).to(self.device)
        dataset = TensorDataset(tensor_x, tensor_x)  # 自编码器方式
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def fit(self, X: np.ndarray):
        """训练特征提取器"""
        self.model.train()
        train_loader = self._prepare_data(X)

        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x, _ in train_loader:
                self.optimizer.zero_grad()

                # 前向传播：编码和解码
                reconstructed = self.model(batch_x)

                # 计算重构损失
                loss = F.mse_loss(reconstructed, batch_x)

                # 反向传播
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            self.scheduler.step(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.4f}')

    def transform(self, X: np.ndarray) -> np.ndarray:
        """提取深度特征"""
        self.model.eval()
        loader = self._prepare_data(X)
        features = []

        with torch.no_grad():
            for batch_x, _ in loader:
                # 仅提取编码器输出
                encoded_features = self.model.encoder(batch_x)
                features.append(encoded_features.cpu().numpy())
        result = np.concatenate(features, axis=0)
        if result.shape[0] != X.shape[0]:
            raise ValueError(f"特征提取后样本数不匹配: 输入 {X.shape[0]}, 输出 {result.shape[0]}")
        return result


class MLPipeline:
    """
    自动化机器学习流水线，支持分类和回归任务
    """

    def __init__(self, task_type: str = 'classification',
                 random_state: int = 42,
                 device: str = None):
        self.task_type = task_type
        self.random_state = random_state
        self.device = self._init_device(device)
        self.logger = self._init_logger()
        self.models = self._init_models()
        self.results = {}
        self.config = None
        self.feature_extractor = None
        self.best_model = None  # 添加一个属性来存储最优模型
        self.memory_monitor = MemoryMonitor(threshold_mb=1000)
        self.training_monitor = None
        self.cv_tracker = CrossValidationTracker(logger=self.logger)
        self.checkpoint_manager = None
        self.data_processor = DataTypeProcessor(logger=self.logger)

    def preprocess_with_monitoring(self,
                                   df: pd.DataFrame,
                                   target_col: str) -> Tuple[pd.DataFrame, pd.Series, Dict,Dict]:
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

        return X, y, memory_profile,type_report

    def train_with_monitoring(self,
                              model,
                              X_train: pd.DataFrame,
                              y_train: pd.Series,
                              X_val: pd.DataFrame,
                              y_val: pd.Series,
                              config: Dict) -> Dict:
        """带监控的模型训练"""
        # 初始化训练监控器
        self.training_monitor = ModelTrainingMonitor(
            patience=config.get('early_stopping_patience', 10),
            min_delta=config.get('min_delta', 1e-4),
            monitor=config.get('monitor_metric', 'val_loss'),
            logger=self.logger
        )

        # 初始化检查点管理器
        self.checkpoint_manager = ModelCheckpoint(
            save_dir=config.get('checkpoint_dir', 'checkpoints'),
            monitor=config.get('monitor_metric', 'val_loss'),
            mode=config.get('monitor_mode', 'min'),
            logger=self.logger
        )

        # 开始训练监控
        self.training_monitor.start_monitoring()

        # 训练循环
        epochs = config.get('epochs', 100)
        for epoch in range(epochs):
            # 训练逻辑
            model.fit(X_train, y_train)

            # 计算指标
            train_metrics = self._compute_metrics(model, X_train, y_train)
            val_metrics = self._compute_metrics(model, X_val, y_val)

            metrics = {
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'train_accuracy': train_metrics.get('accuracy', 0),
                'val_accuracy': val_metrics.get('accuracy', 0)
            }

            # 更新监控器
            if self.training_monitor(epoch, metrics):
                self.logger.info("触发早停机制")
                break

            # 保存检查点
            self.checkpoint_manager(epoch, model, metrics)

        return self.training_monitor.get_training_summary()

    def cross_validate_with_tracking(self,
                                     model,
                                     X: pd.DataFrame,
                                     y: pd.Series,
                                     config: Dict) -> Dict:
        """带追踪的交叉验证"""
        cv = config.get('cv', 5)
        model_name = str(type(model).__name__)

        self.memory_monitor.start_monitoring()

        # 执行交叉验证
        for fold, (train_idx, val_idx) in enumerate(KFold(n_splits=cv).split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # 训练模型
            model.fit(X_train, y_train)

            # 评估性能
            metrics = self._compute_metrics(model, X_val, y_val)
            metrics['memory_used'] = self.memory_monitor.check_memory()['memory_used_mb']

            # 记录结果
            self.cv_tracker.record_fold(model_name, fold, metrics)

        # 计算统计信息
        cv_stats = self.cv_tracker.compute_statistics()

        # 记录总结
        self.cv_tracker.log_summary()

        return cv_stats

    def _compute_metrics(self, model, X: pd.DataFrame, y: pd.Series) -> Dict:
        """计算评估指标"""
        metrics = {}

        try:
            y_pred = model.predict(X)

            if self.task_type == 'classification':
                metrics['accuracy'] = accuracy_score(y, y_pred)
                metrics['f1'] = f1_score(y, y_pred, average='weighted')

                try:
                    y_prob = model.predict_proba(X)
                    metrics['auc'] = roc_auc_score(y, y_prob[:, 1])
                except (AttributeError, IndexError):
                    pass

            else:  # regression
                metrics['mse'] = mean_squared_error(y, y_pred)
                metrics['rmse'] = np.sqrt(metrics['mse'])
                metrics['mae'] = mean_absolute_error(y, y_pred)

            # 添加损失值
            metrics['loss'] = metrics.get('mse' if self.task_type == 'regression' else '1-accuracy', 0)

        except Exception as e:
            self.logger.error(f"计算指标时出错: {str(e)}")
            metrics['error'] = str(e)

        return metrics

    def setup_config(self, X, y, user_config: Optional[Dict] = None):
        """设置pipeline配置"""
        self.config = get_pipeline_config(X, y, user_config)
        self.logger.info("Pipeline配置已更新")
        self.logger.info(f"配置详情: {self.config}")
        return self.config

    def _init_device(self, device: Optional[str] = None) -> str:
        print(f"CUDA是否可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"当前GPU设备: {torch.cuda.get_device_name(0)}")

        """初始化计算设备"""
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return device

    def _init_logger(self) -> logging.Logger:
        """初始化日志记录器"""
        logger = logging.getLogger('MLPipeline')
        logger.setLevel(logging.INFO)
        # 清理已有处理器，避免重复添加
        for handler in logger.handlers[:]:  # 复制处理器列表进行迭代
            logger.removeHandler(handler)
        # 添加新的处理器
        if not logger.handlers:  # 确保没有处理器后添加
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _init_models(self) -> Dict:
        """初始化模型字典"""
        models = {}
        if self.task_type == 'classification':
            models = {
                'logistic': LogisticRegression(random_state=self.random_state),
                'rf': RandomForestClassifier(random_state=self.random_state),
                'xgb': xgb.XGBClassifier(random_state=self.random_state),
                'lgb': lgb.LGBMClassifier(random_state=self.random_state),
                'catboost': cb.CatBoostClassifier(random_state=self.random_state)
            }
        else:
            models = {
                'linear': LinearRegression(),
                'ridge': Ridge(random_state=self.random_state),
                'lasso': Lasso(random_state=self.random_state),
                'rf': RandomForestRegressor(random_state=self.random_state),
                'xgb': xgb.XGBRegressor(random_state=self.random_state)
            }
        return models

    def load_data(self,
                  data_dir: str,
                  target_col: str,
                  test_size: float = 0.2,
                  stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        加载数据并进行训练集验证集拆分
        """
        train_files = [f for f in os.listdir(data_dir) if f.startswith('train')]
        if not train_files:
            raise FileNotFoundError(f"未找到训练数据文件。目录 {data_dir} 中的文件有: {os.listdir(data_dir)}")

        train_file = train_files[0]
        self.logger.info(f"正在读取训练文件: {train_file}")
        file_ext = os.path.splitext(train_file)[1]
        file_path = os.path.join(data_dir, train_file)

        try:
            if file_ext == '.csv':
                df = pd.read_csv(file_path)
            elif file_ext in ['.xls', '.xlsx']:
                df = pd.read_excel(file_path)
            elif file_ext == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                raise ValueError(f"不支持的文件格式: {file_ext}")

            # 检查数据完整性
            if df.empty:
                raise ValueError("加载的数据集为空")

            # 检查目标列
            if target_col not in df.columns:
                raise ValueError(f"目标列 '{target_col}' 不在数据集中。可用的列有: {df.columns.tolist()}")

            # 分离特征和标签
            X = df.drop(columns=[target_col])
            y = df[target_col]

            # 初始化训练集变量（修复之前的bug）
            X_train = X
            y_train = y

            # 处理验证集
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
                    test_size=test_size,
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

    def analyze_data(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Dict:
        """
        数据分析函数

        Args:
            df: 待分析的数据框
            target_col: 目标变量列名
        """
        self.logger.info(f"开始数据分析，数据形状: {df.shape}")
        self.logger.info(f"数据列名: {df.columns.tolist()}")

        if target_col and target_col not in df.columns:
            raise ValueError(f"目标列 '{target_col}' 不在数据集中。可用的列有: {df.columns.tolist()}")

        analysis_results = {}

        # 基本统计信息
        analysis_results['basic_stats'] = df.describe()

        # 区分特征类型
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        analysis_results['feature_types'] = {
            'numeric': list(numeric_cols),
            'categorical': list(categorical_cols)
        }

        # 空值分析
        null_analysis = df.isnull().mean() * 100
        analysis_results['null_percentage'] = null_analysis

        # 如果是分类任务，分析目标变量分布
        if target_col and self.task_type == 'classification':
            class_distribution = df[target_col].value_counts(normalize=True)
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

        return analysis_results

    def setup_feature_extractor(self, input_dim: int, config: Dict):
        """设置深度特征提取器"""
        self.feature_extractor = DeepFeatureExtraction(
            input_dim=input_dim,
            hidden_dims=config.get('hidden_dims', [256, 128, 64]),
            batch_size=config.get('batch_size', 32),
            epochs=config.get('epochs', 50),
            learning_rate=config.get('learning_rate', 1e-3),
            dropout_rate=config.get('dropout_rate', 0.3),
            device=self.device
        )

    def extract_deep_features(self, X: np.ndarray) -> np.ndarray:
        """使用深度学习模型提取特征"""
        if self.feature_extractor is None:
            raise ValueError("Feature extractor not initialized. Call setup_feature_extractor first.")

        return self.feature_extractor.transform(X)

    def preprocess_features(self,
                            X_train: pd.DataFrame,
                            X_val: pd.DataFrame,
                            y_train: pd.Series = None,
                            y_val: pd.Series = None,
                            config: Dict = {}) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        完整的特征预处理方法，包含基础预处理和深度学习特征提取
        """
        X_train_processed = X_train.copy()
        X_val_processed = X_val.copy()
        y_train_processed = y_train.copy() if y_train is not None else None
        y_val_processed = y_val.copy() if y_val is not None else None

        # 在每个重要步骤后添加数据完整性检查
        self._check_data_consistency("Initial state", X_train_processed, y_train_processed)

        # 1. 基础预处理部分
        categorical_features = X_train.select_dtypes(include=['object', 'category']).columns

        # 1.1 处理分类特征
        encoders = {}
        for col in categorical_features:
            if config.get('encoding_method', 'label') == 'label':
                encoder = LabelEncoder()
                X_train_processed[col] = encoder.fit_transform(X_train_processed[col])
                X_val_processed[col] = encoder.transform(X_val_processed[col])
            else:  # onehot encoding
                encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                col_encoded = encoder.fit_transform(X_train_processed[[col]])
                col_encoded_val = encoder.transform(X_val_processed[[col]])

                new_cols = [f"{col}_{i}" for i in range(col_encoded.shape[1])]

                for i, new_col in enumerate(new_cols):
                    X_train_processed[new_col] = col_encoded[:, i]
                    X_val_processed[new_col] = col_encoded_val[:, i]

                X_train_processed = X_train_processed.drop(columns=[col])
                X_val_processed = X_val_processed.drop(columns=[col])

            encoders[col] = encoder

        self._check_data_consistency("After encoding", X_train_processed, y_train_processed)

        # 1.2 空值处理
        numeric_cols = X_train_processed.select_dtypes(include=['int64', 'float64']).columns
        if config.get('handle_missing', True):
            imputer = SimpleImputer(strategy="mean")
            if len(numeric_cols) > 0:
                X_train_processed[numeric_cols] = imputer.fit_transform(X_train_processed[numeric_cols])
                X_val_processed[numeric_cols] = imputer.transform(X_val_processed[numeric_cols])

        self._check_data_consistency("After imputation", X_train_processed, y_train_processed)

        # 1.3 特征缩放
        if config.get('scale_features', True):
            scaler = StandardScaler() if config.get('scaler', 'standard') == 'standard' else MinMaxScaler()
            if len(numeric_cols) > 0:
                X_train_processed[numeric_cols] = scaler.fit_transform(X_train_processed[numeric_cols])
                X_val_processed[numeric_cols] = scaler.transform(X_val_processed[numeric_cols])

        self._check_data_consistency("After scaling", X_train_processed, y_train_processed)

        # 2. 深度学习特征提取部分
        if config.get('use_deep_features', False):
            try:
                if self.feature_extractor is None:
                    self.setup_feature_extractor(
                        input_dim=X_train_processed.shape[1],
                        config=config.get('deep_feature_config', {})
                    )
                    self.feature_extractor.fit(X_train_processed.values)

                deep_features_train = self.feature_extractor.transform(X_train_processed.values)
                deep_features_val = self.feature_extractor.transform(X_val_processed.values)

                deep_feature_cols = [f'deep_feature_{i}' for i in range(deep_features_train.shape[1])]

                # 确保深度特征的样本数量匹配
                if len(deep_features_train) != len(X_train_processed):
                    raise ValueError(f"深度特征样本数不匹配: {len(deep_features_train)} vs {len(X_train_processed)}")

                X_train_processed = pd.concat([
                    X_train_processed,
                    pd.DataFrame(deep_features_train, columns=deep_feature_cols, index=X_train_processed.index)
                ], axis=1)

                X_val_processed = pd.concat([
                    X_val_processed,
                    pd.DataFrame(deep_features_val, columns=deep_feature_cols, index=X_val_processed.index)
                ], axis=1)

                self._check_data_consistency("After deep features", X_train_processed, y_train_processed)

            except Exception as e:
                self.logger.error(f"深度特征提取失败: {str(e)}")
                # 继续处理，但不使用深度特征

        # 3. 特征选择
        if config.get('feature_selection', {}).get('enabled', False):
            try:
                feature_selection_config = config.get('feature_selection', {})
                n_features = min(
                    feature_selection_config.get('n_features', X_train_processed.shape[1] // 2),
                    X_train_processed.shape[1]
                )

                # 确保数据没有无穷值或NaN
                X_train_processed = X_train_processed.replace([np.inf, -np.inf], np.nan)
                X_val_processed = X_val_processed.replace([np.inf, -np.inf], np.nan)

                # 再次进行空值填充
                imputer = SimpleImputer(strategy="mean")
                X_train_processed = pd.DataFrame(
                    imputer.fit_transform(X_train_processed),
                    columns=X_train_processed.columns,
                    index=X_train_processed.index
                )
                X_val_processed = pd.DataFrame(
                    imputer.transform(X_val_processed),
                    columns=X_val_processed.columns,
                    index=X_val_processed.index
                )

                # 确保所有值都是非负的（对于chi2）
                if self.task_type == 'classification':
                    min_values = X_train_processed.min()
                    X_train_processed = X_train_processed - min_values + 0.1
                    X_val_processed = X_val_processed - min_values + 0.1
                    selector = SelectKBest(score_func=chi2, k=n_features)
                else:
                    selector = SelectKBest(score_func=mutual_info_classif, k=n_features)

                # 在特征选择之前再次检查数据一致性
                self._check_data_consistency("Before feature selection", X_train_processed, y_train_processed)

                X_train_selected = selector.fit_transform(X_train_processed.values, y_train_processed)
                X_val_selected = selector.transform(X_val_processed.values)

                # 获取选中的特征名称
                selected_features = X_train_processed.columns[selector.get_support()].tolist()

                X_train_processed = pd.DataFrame(
                    X_train_selected,
                    columns=selected_features,
                    index=X_train_processed.index
                )
                X_val_processed = pd.DataFrame(
                    X_val_selected,
                    columns=selected_features,
                    index=X_val_processed.index
                )

                self._check_data_consistency("After feature selection", X_train_processed, y_train_processed)

            except Exception as e:
                self.logger.error(f"特征选择失败: {str(e)}")
                # 继续处理，但不进行特征选择

        # 最终的数据一致性检查
        self._check_data_consistency("Final state", X_train_processed, y_train_processed)

        return X_train_processed, X_val_processed, y_train_processed, y_val_processed

    def _check_data_consistency(self, stage: str, X: pd.DataFrame, y: pd.Series):
        """检查数据一致性"""
        if y is not None and len(X) != len(y):
            raise ValueError(
                f"在{stage}阶段检测到数据不一致："
                f"X shape: {X.shape}, "
                f"y shape: {y.shape if hasattr(y, 'shape') else len(y)}"
            )
        self.logger.info(f"数据一致性检查通过 - {stage}: X shape {X.shape}")

    def train_model(self,
                    X_train: pd.DataFrame,
                    y_train: pd.Series,
                    X_val: pd.DataFrame,
                    y_val: pd.Series,
                    model_name: str,
                    params: Optional[Dict] = None) -> Dict:
        """
        模型训练与评估

        Args:
            X_train: 训练集特征
            y_train: 训练集标签
            X_val: 验证集特征
            y_val: 验证集标签
            model_name: 模型名称
            params: 模型参数

        Returns:
            包含训练结果的字典
        """

        model = self.models[model_name]
        gpu_available = torch.cuda.is_available()

        # 检查是否安装了 xgboost-gpu 或 lightgbm-gpu
        if model_name == 'xgb' and gpu_available:
            try:
                params['tree_method'] = 'hist'  # 使用GPU版本
                params['device'] = 'cuda'
            except ImportError:
                pass  # 如果没有安装xgboost-gpu,就默认使用CPU版本
        elif model_name == 'lgb' and gpu_available:
            try:
                params['device'] = 'gpu'  # 使用GPU版本
            except ImportError:
                pass  # 如果没有安装lightgbm-gpu,就默认使用CPU版本
        elif model_name == 'lgb' and gpu_available:
            try:
                params['task_type'] = 'GPU'  # 使用GPU版本
            except ImportError:
                pass  # 如果没有安装lightgbm-gpu,就默认使用CPU版本

        if params is None and model_name in self.results and 'best_params' in self.results[model_name]:
            params = self.results[model_name]['best_params']

        if params:
            model.set_params(**params)

        # 训练模型
        model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_val)
        y_pred_train = model.predict(X_train)

        # 评估结果
        results = {
            'model_name': model_name,
            'train_size': len(X_train),
            'val_size': len(X_val)
        }

        # 根据任务类型计算评估指标
        if self.task_type == 'classification':
            results.update({
                'train_accuracy': accuracy_score(y_train, y_pred_train),
                'val_accuracy': accuracy_score(y_val, y_pred),
                'train_f1': f1_score(y_train, y_pred_train, average='weighted'),
                'val_f1': f1_score(y_val, y_pred, average='weighted'),
                'train_recall': recall_score(y_train, y_pred_train, average='weighted'),
                'val_recall': recall_score(y_val, y_pred, average='weighted')
            })

            # 对于二分类问题，计算 ROC-AUC
            if len(np.unique(y_train)) == 2:
                try:
                    y_pred_proba = model.predict_proba(X_val)[:, 1]
                    results['val_roc_auc'] = roc_auc_score(y_val, y_pred_proba)
                except (AttributeError, IndexError):
                    self.logger.warning(f"{model_name} 不支持概率预测，跳过 ROC-AUC 计算")
        else:
            results.update({
                'train_mse': mean_squared_error(y_train, y_pred_train),
                'val_mse': mean_squared_error(y_val, y_pred),
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'val_rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
                'train_mae': mean_absolute_error(y_train, y_pred_train),
                'val_mae': mean_absolute_error(y_val, y_pred)
            })

        return results

    def select_best_model(self, results: Dict) -> str:
        # 添加: 选择效果最好的模型
        best_model = None
        best_score = 0
        for model, result in results.items():
            if 'val_accuracy' in result and result['val_accuracy'] > best_score:
                best_model = model
                best_score = result['val_accuracy']
            elif 'val_f1' in result and result['val_f1'] > best_score:
                best_model = model
                best_score = result['val_f1']
        return best_model

    def optimize_hyperparameters(self,
                                 X_train: pd.DataFrame,
                                 y_train: pd.Series,
                                 model_name: str,
                                 param_grid: Dict,
                                 cv: int = 5,
                                 n_trials: int = 100,
                                 optimization_method: str = 'optuna') -> Dict:
        """优化超参数"""
        self.logger.info(f"开始 {model_name} 模型的超参数优化")

        if not param_grid:
            self.logger.warning(f"未提供参数网格，将使用默认参数")
            return {}

        model = self.models[model_name]
        gpu_available = torch.cuda.is_available()

        def custom_cross_val_score(model, X, y, cv_splits):
            """自定义交叉验证评分函数，支持GPU和常规模型"""
            scores = []
            for train_idx, val_idx in cv_splits:
                X_train_cv = X.iloc[train_idx]
                y_train_cv = y.iloc[train_idx]
                X_val_cv = X.iloc[val_idx]
                y_val_cv = y.iloc[val_idx]

                try:
                    # 常规模型训练（包括逻辑回归和随机森林）
                    if isinstance(model, (LogisticRegression, RandomForestClassifier)) or \
                            not hasattr(model, 'to'):
                        model.fit(X_train_cv, y_train_cv)
                        y_pred = model.predict(X_val_cv)
                        score = accuracy_score(y_val_cv, y_pred)

                    # GPU加速模型训练
                    elif model_name in ['xgb', 'lgb', 'catboost']:
                        model.fit(X_train_cv, y_train_cv,
                                  eval_set=[(X_val_cv, y_val_cv)],
                                  early_stopping_rounds=10,
                                  verbose=False)
                        y_pred = model.predict(X_val_cv)
                        score = accuracy_score(y_val_cv, y_pred)

                    # PyTorch模型训练
                    elif hasattr(model, 'to'):
                        X_train_gpu = torch.tensor(X_train_cv.values, dtype=torch.float32).to(self.device)
                        y_train_gpu = torch.tensor(y_train_cv.values, dtype=torch.float32).to(self.device)
                        X_val_gpu = torch.tensor(X_val_cv.values, dtype=torch.float32).to(self.device)
                        y_val_gpu = torch.tensor(y_val_cv.values, dtype=torch.float32).to(self.device)

                        model.train()
                        optimizer = torch.optim.Adam(model.parameters())
                        criterion = torch.nn.CrossEntropyLoss()

                        for _ in range(10):
                            optimizer.zero_grad()
                            outputs = model(X_train_gpu)
                            loss = criterion(outputs, y_train_gpu.long())
                            loss.backward()
                            optimizer.step()

                        model.eval()
                        with torch.no_grad():
                            outputs = model(X_val_gpu)
                            _, predicted = torch.max(outputs.data, 1)
                            score = accuracy_score(y_val_gpu.cpu().numpy(), predicted.cpu().numpy())

                    scores.append(score)

                except Exception as e:
                    self.logger.warning(f"交叉验证过程中发生错误: {str(e)}")
                    continue

            if not scores:
                raise ValueError("所有交叉验证折都失败了")

            return np.array(scores)

        if optimization_method == 'optuna':
            import optuna

            def objective(trial):
                params = {}
                # 根据参数类型动态生成搜索空间
                for param_name, param_range in param_grid.items():
                    if param_name in ['tree_method', 'device', 'task_type', 'predictor', 'gpu_id']:
                        continue
                    if isinstance(param_range[0], bool):
                        params[param_name] = trial.suggest_categorical(param_name, param_range)
                    elif isinstance(param_range[0], int):
                        params[param_name] = trial.suggest_int(param_name,
                                                               min(param_range),
                                                               max(param_range))
                    elif isinstance(param_range[0], float):
                        params[param_name] = trial.suggest_float(param_name,
                                                                 min(param_range),
                                                                 max(param_range),
                                                                 log=True)
                    else:
                        params[param_name] = trial.suggest_categorical(param_name, param_range)

                try:
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")

                        # 初始化当前trial的模型
                        if model_name == 'xgb':
                            current_model = xgb.XGBClassifier(
                                tree_method='gpu_hist',
                                predictor='gpu_predictor',
                                gpu_id=0,
                                **params
                            )
                        elif model_name == 'lgb':
                            current_model = lgb.LGBMClassifier(
                                device='gpu',
                                gpu_platform_id=0,
                                gpu_device_id=0,
                                **params
                            )
                        elif model_name == 'catboost':
                            current_model = cb.CatBoostClassifier(
                                task_type='GPU',
                                devices='0',
                                **params
                            )
                        elif model_name == 'logistic':
                            current_model = LogisticRegression(**params)
                        elif model_name == 'rf':
                            # 随机森林的特殊处理
                            if 'max_depth' in params and params['max_depth'] is None:
                                params['max_depth'] = None  # 确保None值正确传递
                            current_model = RandomForestClassifier(random_state=self.random_state,
                                                                   n_jobs=-1,  # 使用所有CPU核心
                                                                   **params)
                        else:
                            current_model = model.set_params(**params)

                        # 准备交叉验证
                        cv_splits = list(
                            KFold(n_splits=cv, shuffle=True, random_state=self.random_state).split(X_train))

                        # 使用自定义的交叉验证评分函数
                        scores = custom_cross_val_score(current_model, X_train, y_train, cv_splits)

                        mean_score = np.mean(scores)
                        std_score = np.std(scores)

                        # 检查警告
                        has_convergence_warning = any("convergence" in str(warn.message) for warn in w)

                        # 记录当前试验
                        self.logger.info(
                            f"\n试验 {trial.number}:"
                            f"\n参数: {params}"
                            f"\n平均得分: {mean_score:.4f} (±{std_score:.4f})"
                            f"\n收敛警告: {'是' if has_convergence_warning else '否'}"
                        )

                        if has_convergence_warning:
                            mean_score *= 0.95

                        return mean_score

                except Exception as e:
                    self.logger.error(f"试验失败: {str(e)}")
                    raise  # 抛出异常以便optuna处理

            # 创建和运行优化研究
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=self.random_state)
            )

            try:
                study.optimize(
                    objective,
                    n_trials=n_trials,
                    n_jobs=1,  # 使用GPU时避免并行
                    show_progress_bar=True
                )

                best_params = study.best_params
                best_score = study.best_value

                # 记录最佳结果
                self.logger.info(f"\n最佳参数: {best_params}")
                self.logger.info(f"最佳得分: {best_score:.4f}")

            except Exception as e:
                self.logger.error(f"优化过程失败: {str(e)}")
                raise

        else:
            # 网格搜索实现
            cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=cv_splitter,
                n_jobs=-1 if model_name == 'rf' else 1,  # RF可以使用多核
                scoring='accuracy' if self.task_type == 'classification' else 'neg_mean_squared_error',
                verbose=2,
                refit=True
            )

            try:
                grid_search.fit(X_train, y_train)
                best_params = grid_search.best_params_
                best_score = grid_search.best_score_

            except Exception as e:
                self.logger.error(f"网格搜索失败: {str(e)}")
                raise

        # 确保GPU相关设置保持不变
        if model_name == 'xgb':
            best_params['tree_method'] = 'gpu_hist'
            best_params['predictor'] = 'gpu_predictor'
        elif model_name == 'lgb':
            best_params['device'] = 'gpu'
        elif model_name == 'catboost':
            best_params['task_type'] = 'GPU'

        return best_params

    def save_model(self, model: object, filepath: str) -> None:
        """
        保存模型

        Args:
            model: 训练好的模型对象
            filepath: 保存路径
        """
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            self.logger.info(f"模型已保存至: {filepath}")
        except Exception as e:
            self.logger.error(f"模型保存失败: {str(e)}")

    def load_model(self, filepath: str) -> object:
        """
        加载模型

        Args:
            filepath: 模型文件路径

        Returns:
            加载的模型对象
        """
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            self.logger.info(f"模型已从 {filepath} 加载")
            return model
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")
            return None

    def check_data_integrity(self, df: pd.DataFrame, stage: str = "unknown"):
        """
        检查数据完整性
        """
        self.logger.info(f"=== 数据完整性检查 (阶段: {stage}) ===")
        self.logger.info(f"数据形状: {df.shape}")
        self.logger.info(f"列名: {df.columns.tolist()}")
        self.logger.info(f"数据类型:\n{df.dtypes}")

    def run_pipeline(self,
                     data_dir: str,
                     target_col: str,
                     output_dir: str,
                     config: Dict) -> Dict:
        """运行完整的机器学习流水线"""
        # 初始化监控组件（如果还没有初始化）
        if not hasattr(self, 'memory_monitor'):
            self.memory_monitor = MemoryMonitor(threshold_mb=1000)
        if not hasattr(self, 'data_processor'):
            self.data_processor = DataTypeProcessor(logger=self.logger)

        # 开始内存监控
        self.memory_monitor.start_monitoring()

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 加载数据
        X_train, X_val, y_train, y_val = self.load_data(
            data_dir=data_dir,
            target_col=target_col,
            test_size=config.get('test_size', 0.2),
            stratify=config.get('stratify', True)
        )

        # 设置并获取配置
        pipeline_config = self.setup_config(X_train, y_train, config)

        # 数据分析阶段（带内存监控）
        self.logger.info("开始数据分析阶段...")
        train_data = X_train.copy()
        train_data[target_col] = y_train
        val_data = X_val.copy()
        val_data[target_col] = y_val
        complete_data = pd.concat([train_data, val_data])

        memory_stats = self.memory_monitor.check_memory()
        self.logger.info(f"数据合并后内存使用: {memory_stats['memory_used_mb']:.2f}MB")

        analysis_results = self.analyze_data(
            df=complete_data,
            target_col=target_col
        )

        # # 特征预处理阶段（带类型优化）
        # self.logger.info("开始特征预处理阶段...")
        # # 首先进行数据类型优化
        # train_data_optimized = self.data_processor.infer_and_convert_types(complete_data)
        # type_report = self.data_processor.check_data_types(train_data_optimized)
        # self.logger.info(f"数据类型优化报告: {type_report}")

        X, y, memory_profile,type_report=self.preprocess_with_monitoring(df=complete_data,target_col=target_col)

        X_train_processed, X_val_processed, y_train_processed, y_val_processed = self.preprocess_features(
            X_train=X,
            X_val=X_val,
            y_train=y,
            y_val=y_val,
            config=pipeline_config.preprocess.__dict__
        )

        # 模型训练与评估
        results = {}
        models_to_train = config.get('models', list(self.models.keys()))
        self.logger.info(f"将训练以下模型: {models_to_train}")

        for model_name in models_to_train:
            self.logger.info(f"\n开始训练模型: {model_name}")
            memory_before = self.memory_monitor.check_memory(log_usage=True)
            # 初始化训练监控器
            self.training_monitor = ModelTrainingMonitor(
                patience=config.get('early_stopping_patience', 10),
                min_delta=config.get('min_delta', 1e-4),
                monitor='val_loss',
                logger=self.logger
            )

            # 检查是否需要进行超参数优化
            if config.get('optimize_hyperparameters', False):
                self.logger.info(f"开始 {model_name} 的超参数优化...")
                param_grid = config.get('param_grids', {}).get(model_name, {})
                if not param_grid:
                    self.logger.warning(f"未找到 {model_name} 的参数网格，将使用默认参数")
                    best_params = None
                else:
                    try:
                        best_params = self.optimize_hyperparameters(
                            X_train=X_train_processed,
                            y_train=y_train_processed,
                            model_name=model_name,
                            param_grid=param_grid,
                            cv=config.get('cv', 5),
                            n_trials=config.get('n_trials', 100),
                            optimization_method=config.get('optimization_method', 'grid')
                        )
                        self.logger.info(f"{model_name} 的最佳参数: {best_params}")
                        if best_params:
                            results[model_name] = {'best_params': best_params}
                    except Exception as e:
                        self.logger.error(f"{model_name} 超参数优化失败: {str(e)}")
                        best_params = None
            else:
                self.logger.info("跳过超参数优化")
                best_params = None

            # 训练和评估模型（带监控）
            try:
                self.training_monitor.start_monitoring()
                model_results = self.train_model(
                    X_train=X_train_processed,
                    y_train=y_train_processed,
                    X_val=X_val_processed,
                    y_val=y_val_processed,
                    model_name=model_name,
                    params=best_params
                )

                # 确保模型结果是字典类型
                if not isinstance(model_results, dict):
                    model_results = {'result': model_results}

                results[model_name] = model_results

                # 添加训练监控结果
                model_results['training_summary'] = self.training_monitor.get_training_summary()
                results[model_name].update(model_results)

                memory_after = self.memory_monitor.check_memory(log_usage=True)
                results[model_name]['memory_usage'] = {
                    'before_training': memory_before,
                    'after_training': memory_after,
                    'memory_increase': memory_after['memory_used_mb'] - memory_before['memory_used_mb']
                }
                # 保存模型
                if config.get('save_models', False):
                    model_path = os.path.join(
                        output_dir,
                        f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                    )
                    self.save_model(self.models[model_name], model_path)

            except Exception as e:
                self.logger.error(f"{model_name} 训练失败: {str(e)}")
                results[model_name] = {"error": str(e)}

        # 保存分析结果
        analysis_path = os.path.join(output_dir, 'analysis_results.pkl')
        with open(analysis_path, 'wb') as f:
            pickle.dump(analysis_results, f)

        # 记录最终的内存使用情况
        final_memory_profile = self.memory_monitor.get_memory_profile()
        self.logger.info(f"内存使用概况: {final_memory_profile}")

        # 输出训练结果
        self.logger.info("\n训练完成！最终结果:")
        for model_name, model_results in results.items():
            if "error" not in model_results:
                self.logger.info(f"\n{model_name} 模型性能:")
                for metric, value in model_results.items():
                    if isinstance(value, (int, float)):
                        self.logger.info(f"{metric}: {value:.4f}")

        self.best_model = self.select_best_model(results)
        results['best_model'] = self.best_model

        return {
            'analysis_results': analysis_results,
            'model_results': results,
            'final_config': pipeline_config,
            'memory_profile': final_memory_profile,
            'data_type_report': type_report
        }


class DataTypeProcessor:
    """数据类型处理器"""

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        # 定义可接受的数据类型映射
        self.dtype_mappings = {
            'int': ['int8', 'int16', 'int32', 'int64'],
            'float': ['float16', 'float32', 'float64'],
            'category': ['object', 'category'],
            'datetime': ['datetime64[ns]', 'datetime64']
        }

        # 内存优化的类型映射
        self.memory_efficient_types = {
            'int': {
                'small': 'int8',
                'medium': 'int16',
                'large': 'int32',
                'very_large': 'int64'
            },
            'float': {
                'small': 'float32',
                'large': 'float64'
            }
        }

    def infer_and_convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """推断并转换数据类型"""
        df = df.copy()

        for column in df.columns:
            try:
                # 检查是否为数值型
                if df[column].dtype in self.dtype_mappings['int'] + self.dtype_mappings['float']:
                    # 检查是否有空值
                    if df[column].isna().any():
                        self.logger.info(f"列 {column} 包含空值，保持原有类型")
                        continue

                    # 检查是否有无限值
                    if np.isinf(df[column]).any():
                        self.logger.info(f"列 {column} 包含无限值，保持原有类型")
                        continue

                    df[column] = self._optimize_numeric_type(df[column])

                # 检查是否为日期型
                elif self._is_datetime(df[column]):
                    try:
                        df[column] = pd.to_datetime(df[column], errors='coerce')
                    except Exception as e:
                        self.logger.warning(f"列 {column} 转换为日期类型失败: {str(e)}")

                # 检查是否为分类型
                elif self._should_be_category(df[column]):
                    if df[column].isna().sum() / len(df[column]) < 0.05:  # 空值比例小于5%
                        df[column] = df[column].astype('category')

                self.logger.info(f"列 {column} 类型已优化为: {df[column].dtype}")

            except Exception as e:
                self.logger.warning(f"列 {column} 类型转换失败: {str(e)}")
                # 保持原有类型
                continue

        return df

    def _optimize_numeric_type(self, series: pd.Series) -> pd.Series:
        """优化数值类型"""
        if series.dtype in self.dtype_mappings['int']:
            max_val = series.max()
            min_val = series.min()

            # 根据数值范围选择最优的整数类型
            if min_val >= -128 and max_val <= 127:
                return series.astype('int8')
            elif min_val >= -32768 and max_val <= 32767:
                return series.astype('int16')
            elif min_val >= -2147483648 and max_val <= 2147483647:
                return series.astype('int32')
            else:
                return series.astype('int64')

        elif series.dtype in self.dtype_mappings['float']:
            # 检查是否可以转换为整数
            if series.equals(series.astype(int)):
                return self._optimize_numeric_type(series.astype(int))
            else:
                # 根据精度需求选择float类型
                float_precision = self._calculate_float_precision(series)
                if float_precision <= np.finfo(np.float32).precision:
                    return series.astype('float32')
                return series.astype('float64')

        return series

    def _is_datetime(self, series: pd.Series) -> bool:
        """检查是否为日期时间类型"""
        if series.dtype in self.dtype_mappings['datetime']:
            return True

        try:
            pd.to_datetime(series.iloc[0])
            return True
        except:
            return False

    def _should_be_category(self, series: pd.Series, threshold: float = 0.05) -> bool:
        """判断是否应该转换为分类型"""
        if series.dtype not in self.dtype_mappings['category']:
            return False

        unique_ratio = series.nunique() / len(series)
        return unique_ratio <= threshold

    def _calculate_float_precision(self, series: pd.Series) -> int:
        """计算浮点数所需精度"""
        decimal_places = series.astype(str).str.extract(r'\.(\d+)')[0].str.len()
        return int(decimal_places.max() or 0)

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

    def _can_downcast_to_int(self, series: pd.Series) -> bool:
        """检查是否可以将浮点数转换为整数"""
        try:
            # 首先检查是否有 NaN 或无限值
            if not np.isfinite(series).all():
                return False

            # 检查是否所有值都等于其整数部分
            return np.allclose(series, series.round(), rtol=1e-10, atol=1e-10)

        except Exception as e:
            self.logger.warning(f"检查整数转换时发生错误: {str(e)}")
            return False

    def optimize_memory_usage(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """优化内存使用"""
        original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        df_optimized = df.copy()

        optimization_report = {
            'original_memory': f"{original_memory:.2f} MB",
            'column_optimizations': {}
        }

        for column in df_optimized.columns:
            original_type = str(df_optimized[column].dtype)
            df_optimized[column] = self._optimize_numeric_type(df_optimized[column])
            new_type = str(df_optimized[column].dtype)

            if original_type != new_type:
                optimization_report['column_optimizations'][column] = {
                    'original_type': original_type,
                    'optimized_type': new_type
                }

        final_memory = df_optimized.memory_usage(deep=True).sum() / 1024 / 1024
        optimization_report['final_memory'] = f"{final_memory:.2f} MB"
        optimization_report['memory_saved'] = f"{(original_memory - final_memory):.2f} MB"
        optimization_report[
            'reduction_percentage'] = f"{((original_memory - final_memory) / original_memory * 100):.2f}%"

        return df_optimized, optimization_report


class ModelTrainingMonitor:
    """模型训练监控器"""

    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 1e-4,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 logger=None):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.logger = logger or logging.getLogger(__name__)

        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.early_stop = False
        self.training_history = []
        self.start_time = None
        self.epoch_times = []

    def __call__(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """
        检查是否应该早停

        Args:
            epoch: 当前epoch
            metrics: 评估指标字典

        Returns:
            bool: 是否应该停止训练
        """
        if self.monitor not in metrics:
            raise ValueError(f"监控指标 {self.monitor} 不在评估指标中")

        score = metrics[self.monitor]
        self.training_history.append(metrics)

        if self.mode == 'min':
            score_improved = score <= (self.best_score - self.min_delta)
        else:
            score_improved = score >= (self.best_score + self.min_delta)

        if score_improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

        # 记录训练进度
        self._log_progress(epoch, metrics)

        # 检查是否应该早停
        if self.counter >= self.patience:
            self.early_stop = True
            self.logger.info(f"触发早停！最佳{self.monitor}: {self.best_score:.4f}")
            return True

        return False

    def start_monitoring(self):
        """开始监控"""
        self.start_time = time.time()

    def _log_progress(self, epoch: int, metrics: Dict[str, float]):
        """记录训练进度"""
        current_time = time.time()
        if self.start_time is None:
            self.start_time = current_time

        elapsed_time = current_time - self.start_time
        self.epoch_times.append(current_time)

        # 计算预估剩余时间
        if len(self.epoch_times) > 1:
            avg_epoch_time = (self.epoch_times[-1] - self.epoch_times[0]) / (len(self.epoch_times) - 1)
            estimated_remaining = avg_epoch_time * (self.patience - epoch)
        else:
            estimated_remaining = 0

        # 格式化输出进度信息
        metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(
            f"Epoch {epoch:3d} - {metrics_str} - "
            f"耗时: {elapsed_time:.2f}s - "
            f"预计剩余: {estimated_remaining:.2f}s"
        )

    def get_training_summary(self) -> Dict:
        """获取训练总结"""
        if not self.training_history:
            return {}

        summary = {
            'best_score': self.best_score,
            'total_epochs': len(self.training_history),
            'early_stopped': self.early_stop,
            'total_time': self.epoch_times[-1] - self.start_time if self.epoch_times else 0,
            'metrics_history': {},
        }

        # 汇总每个指标的历史
        metrics_keys = self.training_history[0].keys()
        for key in metrics_keys:
            values = [epoch_metrics[key] for epoch_metrics in self.training_history]
            summary['metrics_history'][key] = {
                'min': min(values),
                'max': max(values),
                'mean': sum(values) / len(values),
                'std': np.std(values),
                'values': values
            }

        return summary


class ModelCheckpoint:
    """模型检查点"""

    def __init__(self,
                 save_dir: str,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 save_best_only: bool = True,
                 max_saves: int = 3,
                 logger=None):
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.max_saves = max_saves
        self.logger = logger or logging.getLogger(__name__)

        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.saved_models = []

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

    def __call__(self, epoch: int, model: object, metrics: Dict[str, float]) -> None:
        """
        保存模型检查点
        """
        if self.monitor not in metrics:
            raise ValueError(f"监控指标 {self.monitor} 不在评估指标中")

        score = metrics[self.monitor]
        save_path = os.path.join(
            self.save_dir,
            f"model_epoch{epoch:03d}_{self.monitor}{score:.4f}.pkl"
        )

        should_save = not self.save_best_only
        if self.mode == 'min':
            if score <= self.best_score:
                should_save = True
                self.best_score = score
        else:
            if score >= self.best_score:
                should_save = True
                self.best_score = score

        if should_save:
            try:
                with open(save_path, 'wb') as f:
                    pickle.dump(model, f)
                self.saved_models.append(save_path)
                self.logger.info(f"模型已保存至: {save_path}")

                # 如果超过最大保存数量，删除旧的检查点
                if len(self.saved_models) > self.max_saves:
                    oldest_model = self.saved_models.pop(0)
                    if os.path.exists(oldest_model):
                        os.remove(oldest_model)
                        self.logger.info(f"已删除旧的检查点: {oldest_model}")

            except Exception as e:
                self.logger.error(f"保存模型失败: {str(e)}")

    def load_best_model(self) -> Optional[object]:
        """加载最佳模型"""
        if not self.saved_models:
            self.logger.warning("没有可用的模型检查点")
            return None

        try:
            best_model_path = self.saved_models[-1]
            with open(best_model_path, 'rb') as f:
                model = pickle.load(f)
            self.logger.info(f"已加载最佳模型: {best_model_path}")
            return model
        except Exception as e:
            self.logger.error(f"加载模型失败: {str(e)}")
            return None

    def cleanup(self):
        """清理所有保存的模型检查点"""
        for model_path in self.saved_models:
            try:
                if os.path.exists(model_path):
                    os.remove(model_path)
                    self.logger.info(f"已删除检查点: {model_path}")
            except Exception as e:
                self.logger.error(f"删除检查点失败: {str(e)}")
        self.saved_models = []


class CrossValidationTracker:
    """交叉验证结果追踪器"""

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.cv_results = {}
        self.cv_statistics = {}

    def record_fold(self,
                    model_name: str,
                    fold: int,
                    metrics: Dict[str, float],
                    params: Optional[Dict] = None):
        """记录单次交叉验证结果"""
        if model_name not in self.cv_results:
            self.cv_results[model_name] = {
                'folds': {},
                'params': params,
                'timestamps': []
            }

        self.cv_results[model_name]['folds'][fold] = metrics
        self.cv_results[model_name]['timestamps'].append(time.time())

    def compute_statistics(self) -> Dict:
        """计算交叉验证统计信息"""
        for model_name, results in self.cv_results.items():
            fold_metrics = list(results['folds'].values())

            # 计算每个指标的统计量
            stats = {}
            for metric in fold_metrics[0].keys():
                values = [fold[metric] for fold in fold_metrics]
                stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }

            self.cv_statistics[model_name] = stats

        return self.cv_statistics

    def get_best_fold(self, model_name: str, metric: str, mode: str = 'max') -> Tuple[int, float]:
        """获取最佳fold"""
        if model_name not in self.cv_results:
            raise ValueError(f"未找到模型 {model_name} 的交叉验证结果")

        folds = self.cv_results[model_name]['folds']
        fold_scores = [(fold, results[metric]) for fold, results in folds.items()]

        if mode == 'max':
            best_fold, best_score = max(fold_scores, key=lambda x: x[1])
        else:
            best_fold, best_score = min(fold_scores, key=lambda x: x[1])

        return best_fold, best_score

    def log_summary(self):
        """输出交叉验证总结"""
        for model_name, stats in self.cv_statistics.items():
            self.logger.info(f"\n=== {model_name} 交叉验证结果 ===")
            for metric, values in stats.items():
                self.logger.info(
                    f"{metric}:\n"
                    f"  Mean ± Std: {values['mean']:.4f} ± {values['std']:.4f}\n"
                    f"  Range: [{values['min']:.4f}, {values['max']:.4f}]\n"
                    f"  Median: {values['median']:.4f}"
                )


class MemoryMonitor:
    """内存使用监控器"""

    def __init__(self, threshold_mb: float = 1000, logger=None):
        self.threshold_mb = threshold_mb
        self.logger = logger or logging.getLogger(__name__)
        self.memory_usage = []
        self.start_time = None
        self.warnings_count = 0

    def start_monitoring(self):
        """开始监控"""
        self.start_time = time.time()
        self.memory_usage = []
        self.warnings_count = 0

    def check_memory(self, log_usage: bool = True) -> Dict:
        """检查当前内存使用情况"""
        try:
            import psutil
            process = psutil.Process()

            # 获取当前进程的内存使用
            memory_info = process.memory_info()
            current_memory = memory_info.rss / 1024 / 1024  # 转换为MB

            memory_stats = {
                'timestamp': time.time(),
                'memory_used_mb': current_memory,
                'memory_percent': process.memory_percent(),
                'system_total_mb': psutil.virtual_memory().total / 1024 / 1024,
                'system_available_mb': psutil.virtual_memory().available / 1024 / 1024
            }

            self.memory_usage.append(memory_stats)

            # 检查是否超过阈值
            if current_memory > self.threshold_mb:
                self.warnings_count += 1
                self.logger.warning(
                    f"内存使用超过阈值！当前使用: {current_memory:.2f}MB, "
                    f"阈值: {self.threshold_mb}MB"
                )

            if log_usage:
                self.logger.info(
                    f"内存使用: {current_memory:.2f}MB "
                    f"({memory_stats['memory_percent']:.1f}% of system memory)"
                )

            return memory_stats

        except ImportError:
            self.logger.warning("psutil未安装，无法监控内存使用")
            return {}
        except Exception as e:
            self.logger.error(f"内存监控失败: {str(e)}")
            return {}

    def get_memory_profile(self) -> Dict:
        """获取内存使用概况"""
        if not self.memory_usage:
            return {}

        memory_values = [stats['memory_used_mb'] for stats in self.memory_usage]
        peak_memory = max(memory_values)
        peak_time = self.memory_usage[memory_values.index(peak_memory)]['timestamp']

        return {
            'peak_memory_mb': peak_memory,
            'peak_time': time.strftime('%Y-%m-%d %H:%M:%S',
                                       time.localtime(peak_time)),
            'average_memory_mb': np.mean(memory_values),
            'memory_std_mb': np.std(memory_values),
            'total_checks': len(self.memory_usage),
            'warnings_triggered': self.warnings_count,
            'monitoring_duration': time.time() - self.start_time
        }

    def plot_memory_usage(self) -> None:
        """绘制内存使用曲线"""
        try:
            import matplotlib.pyplot as plt

            if not self.memory_usage:
                self.logger.warning("没有可用的内存使用数据")
                return

            times = [(stats['timestamp'] - self.start_time) / 60 for stats in self.memory_usage]
            memory_values = [stats['memory_used_mb'] for stats in self.memory_usage]

            plt.figure(figsize=(10, 6))
            plt.plot(times, memory_values, '-b', label='Memory Usage')
            plt.axhline(y=self.threshold_mb, color='r', linestyle='--',
                        label='Threshold')

            plt.xlabel('Time (minutes)')
            plt.ylabel('Memory Usage (MB)')
            plt.title('Memory Usage Over Time')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.show()

        except ImportError:
            self.logger.warning("matplotlib未安装，无法绘制内存使用曲线")
        except Exception as e:
            self.logger.error(f"绘制内存使用曲线失败: {str(e)}")


# 断点续训支持类
class TrainingCheckpoint:
    """训练断点续训支持"""

    def __init__(self,
                 save_dir: str,
                 max_checkpoints: int = 3,
                 logger=None):
        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.logger = logger or logging.getLogger(__name__)
        self.checkpoints = []

        os.makedirs(save_dir, exist_ok=True)

    def save_checkpoint(self,
                        model_state: Dict,
                        optimizer_state: Dict,
                        epoch: int,
                        metrics: Dict[str, float],
                        additional_info: Optional[Dict] = None) -> str:
        """保存训练检查点"""
        checkpoint = {
            'model_state': model_state,
            'optimizer_state': optimizer_state,
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': time.time()
        }

        if additional_info:
            checkpoint.update(additional_info)

        # 生成检查点文件名
        checkpoint_path = os.path.join(
            self.save_dir,
            f"checkpoint_epoch{epoch:03d}_{time.strftime('%Y%m%d_%H%M%S')}.pt"
        )

        try:
            torch.save(checkpoint, checkpoint_path)
            self.checkpoints.append(checkpoint_path)
            self.logger.info(f"已保存检查点: {checkpoint_path}")

            # 删除旧的检查点
            self._cleanup_old_checkpoints()

            return checkpoint_path

        except Exception as e:
            self.logger.error(f"保存检查点失败: {str(e)}")
            return ""

    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> Optional[Dict]:
        """加载检查点"""
        if checkpoint_path is None:
            if not self.checkpoints:
                self.logger.warning("没有可用的检查点")
                return None
            checkpoint_path = self.checkpoints[-1]

        try:
            checkpoint = torch.load(checkpoint_path)
            self.logger.info(f"已加载检查点: {checkpoint_path}")
            return checkpoint
        except Exception as e:
            self.logger.error(f"加载检查点失败: {str(e)}")
            return None

    def _cleanup_old_checkpoints(self):
        """清理旧的检查点"""
        while len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            try:
                if os.path.exists(old_checkpoint):
                    os.remove(old_checkpoint)
                    self.logger.info(f"已删除旧检查点: {old_checkpoint}")
            except Exception as e:
                self.logger.error(f"删除旧检查点失败: {str(e)}")

    def get_latest_checkpoint(self) -> Optional[str]:
        """获取最新的检查点路径"""
        if not self.checkpoints:
            return None
        return self.checkpoints[-1]

    def cleanup_all(self):
        """清理所有检查点"""
        for checkpoint in self.checkpoints:
            try:
                if os.path.exists(checkpoint):
                    os.remove(checkpoint)
                    self.logger.info(f"已删除检查点: {checkpoint}")
            except Exception as e:
                self.logger.error(f"删除检查点失败: {str(e)}")
        self.checkpoints = []


@dataclass
class ModelConfig:
    """模型配置基类"""
    enabled: bool = True
    params: Dict = field(default_factory=dict)


@dataclass
class DeepLearningConfig(ModelConfig):
    """深度学习模型配置"""
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 100
    dropout_rate: float = 0.3
    batch_norm: bool = True
    early_stopping_patience: int = 10
    min_samples_required: int = 10000  # 最小所需样本数


@dataclass
class TraditionalMLConfig(ModelConfig):
    """传统机器学习模型配置"""
    model_type: str = 'rf'  # rf, xgb, lgb等
    cv_folds: int = 5
    max_samples_for_cv: int = 100000  # 交叉验证的最大样本数


@dataclass
class DataConfig:
    """数据配置"""
    test_size: float = 0.2
    stratify: bool = True
    random_state: int = 42


@dataclass
class PreprocessConfig:
    """预处理配置"""
    handle_missing: bool = True
    numeric_impute_strategy: str = 'mean'
    categorical_impute_strategy: str = 'most_frequent'
    handle_outliers: bool = True
    outlier_method: str = 'isolation_forest'
    outlier_fraction: float = 0.1
    scale_features: bool = True
    scaler: str = 'standard'


@dataclass
class PipelineConfig:
    """Pipeline总配置"""
    data: DataConfig = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    deep_learning: DeepLearningConfig = field(default_factory=DeepLearningConfig)
    traditional_ml: TraditionalMLConfig = field(default_factory=TraditionalMLConfig)

    def __post_init__(self):
        """初始化后的处理"""
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        if isinstance(self.preprocess, dict):
            self.preprocess = PreprocessConfig(**self.preprocess)
        if isinstance(self.deep_learning, dict):
            self.deep_learning = DeepLearningConfig(**self.deep_learning)
        if isinstance(self.traditional_ml, dict):
            self.traditional_ml = TraditionalMLConfig(**self.traditional_ml)


class AutoMLConfig:
    """自动机器学习配置类"""

    def __init__(self):
        self.config = PipelineConfig()

    def analyze_data(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Dict:
        """分析数据特征，返回数据特征字典"""
        n_samples = len(X)
        n_features = X.shape[1]
        n_classes = len(np.unique(y)) if y is not None else None

        return {
            'n_samples': n_samples,
            'n_features': n_features,
            'n_classes': n_classes,
            'memory_usage': X.memory_usage().sum() if isinstance(X, pd.DataFrame) else X.nbytes,
            'data_size_category': self._categorize_data_size(n_samples)
        }

    def _categorize_data_size(self, n_samples: int) -> str:
        """根据样本量分类数据规模"""
        if n_samples < 1000:
            return 'small'
        elif n_samples < 10000:
            return 'medium'
        elif n_samples < 100000:
            return 'large'
        else:
            return 'very_large'

    def auto_configure(self, X: Union[pd.DataFrame, np.ndarray],
                       y: Union[pd.Series, np.ndarray]) -> PipelineConfig:
        """根据数据特征自动配置参数"""
        data_info = self.analyze_data(X, y)
        config = self.config

        # 配置深度学习
        if data_info['n_samples'] >= config.deep_learning.min_samples_required:
            config.deep_learning.enabled = True
            # 根据数据规模调整批次大小
            if data_info['data_size_category'] == 'very_large':
                config.deep_learning.batch_size = 128
            # 根据特征数调整隐藏层
            n_features = data_info['n_features']
            config.deep_learning.hidden_dims = self._auto_hidden_dims(n_features)
        else:
            config.deep_learning.enabled = False

        # 配置传统机器学习
        config.traditional_ml.enabled = True
        if data_info['data_size_category'] == 'very_large':
            # 大数据集使用更快的模型
            config.traditional_ml.model_type = 'lgb'
            # 限制交叉验证的样本数
            config.traditional_ml.max_samples_for_cv = 100000
        else:
            # 小数据集使用更稳定的模型
            config.traditional_ml.model_type = 'rf'

        # 配置预处理
        if data_info['data_size_category'] in ['large', 'very_large']:
            # 大数据集使用更快的预处理方法
            config.preprocess.outlier_method = 'isolation_forest'
            config.preprocess.numeric_impute_strategy = 'median'
        else:
            # 小数据集使用更精确的预处理方法
            config.preprocess.outlier_method = 'lof'
            config.preprocess.numeric_impute_strategy = 'knn'

        return config

    def _auto_hidden_dims(self, n_features: int) -> List[int]:
        """自动计算隐藏层维度"""
        # 使用经验公式：逐层减半，直到64
        dims = []
        current_dim = n_features
        while current_dim > 64:
            current_dim = max(current_dim // 2, 64)
            dims.append(current_dim)
        return dims

    def update_config(self, new_config: Dict) -> None:
        """更新配置"""

        def update_recursive(base_config, new_values):
            if isinstance(new_values, dict):
                for key, value in new_values.items():
                    if hasattr(base_config, key):
                        if isinstance(value, dict) and hasattr(base_config, key):
                            update_recursive(getattr(base_config, key), value)
                        else:
                            setattr(base_config, key, value)
                    else:
                        # 如果属性不存在，直接添加新属性到base_config对象上
                        setattr(base_config, key, value)

        update_recursive(self.config, new_config)


def get_pipeline_config(X: Union[pd.DataFrame, np.ndarray],
                        y: Union[pd.Series, np.ndarray],
                        user_config: Optional[Dict] = None) -> PipelineConfig:
    """获取Pipeline配置的便捷函数"""
    auto_config = AutoMLConfig()
    # 首先进行自动配置
    config = auto_config.auto_configure(X, y)
    # 如果有用户配置，则更新
    if user_config:
        auto_config.update_config(user_config)
    return config


# 使用示例

if __name__ == "__main__":
    import logging
    import os
    import pandas as pd

    # 设置详细的日志级别
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('main')

    # 定义完整的配置字典
    config = {
        # 数据相关配置
        'data': {
            'test_size': 0.2,
            'stratify': True,
            'random_state': 42
        },

        # 预处理配置
        'preprocess': {
            'handle_missing': True,
            'numeric_impute_strategy': 'mean',
            'categorical_impute_strategy': 'most_frequent',
            'missing_strategy': 'knn',
            'knn_neighbors': 5,
            'handle_outliers': True,
            'outlier_method': 'isolation_forest',
            'outlier_fraction': 0.1,
            'scale_features': True,
            'scaler': 'standard',
            'encoding_method': 'onehot',

            # 特征选择配置
            'feature_selection': {
                'enabled': True,
                'n_features': 100
            },

            # 深度特征提取配置
            'use_deep_features': True,
            'deep_feature_config': {
                'hidden_dims': [256, 128, 64],
                'batch_size': 1024,
                'epochs': 10,
                'learning_rate': 1e-3,
                'dropout_rate': 0.3,
                'feature_extraction': {
                    'combine_method': 'concat'  # 可选: concat, add, multiply
                }
            }
        },

        # 深度学习模型配置
        'deep_learning': {
            'enabled': True,
            'hidden_dims': [256, 128, 64],
            'batch_size': 32,
            'learning_rate': 1e-3,
            'epochs': 100,
            'dropout_rate': 0.3,
            'batch_norm': True,
            'early_stopping_patience': 10,
            'min_samples_required': 10000
        },

        # 传统机器学习模型配置
        'traditional_ml': {
            'enabled': True,
            'model_type': 'rf',
            'cv_folds': 5,
            'max_samples_for_cv': 100000
        },

        # 模型训练配置
        #'models': ['logistic', 'rf', 'xgb', 'lgb', 'catboost'],
        'models': ['xgb', 'catboost','logistic'],
        'optimize_hyperparameters': True,
        'optimization_method': 'optuna',
        'n_trials': 10,
        'cv': 5,
        'save_models': True,
        'output_dir': '/kaggle/working/output',
        # 超参数优化的参数网格
        'param_grids': {
            'logistic': {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [1000, 2000, 5000]
            },
            'rf': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'xgb': {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'n_estimators': [100, 200, 300],
                'min_child_weight': [1, 3, 5]
            },
            'lgb': {
                'num_leaves': [31, 63, 127],
                'learning_rate': [0.01, 0.1, 0.3],
                'n_estimators': [100, 200, 300],
                'min_child_samples': [20, 30, 50]
            },
            'catboost': {
                'iterations': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.3],
                'depth': [4, 6, 8],
                'l2_leaf_reg': [1, 3, 5, 7]
            }
        }
    }

    try:
        # 检查数据目录和文件
        data_dir = '/kaggle/input/risk232/'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            logger.warning(f"创建数据目录: {data_dir}")

        # 检查输出目录
        output_dir = config['output_dir']
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"创建输出目录: {output_dir}")

        # 查找训练数据文件
        train_files = [f for f in os.listdir(data_dir) if f.startswith('train')]
        if not train_files:
            raise FileNotFoundError(f"未找到训练数据文件。目录内容: {os.listdir(data_dir)}")

        # 读取并检查数据
        train_file = os.path.join(data_dir, train_files[0])
        logger.info(f"读取训练文件: {train_file}")

        df = pd.read_csv(train_file)
        logger.info(f"原始数据形状: {df.shape}")
        logger.info(f"原始数据列名: {df.columns.tolist()}")

        # 检查目标变量
        target_col = 'loan_status'  # 替换为你的目标变量列名
        if target_col not in df.columns:
            raise ValueError(f"目标变量 '{target_col}' 不在数据列中。可用的列: {df.columns.tolist()}")

        # 初始化并运行pipeline
        pipeline = MLPipeline(
            task_type='classification',  # 或 'regression'
            random_state=42,
            device='cuda'  # 如果有GPU
        )

        # 运行完整流水线
        results = pipeline.run_pipeline(
            data_dir=data_dir,
            target_col=target_col,
            output_dir=output_dir,
            config=config
        )

        # 输出结果
        logger.info("=== 分析结果 ===")
        for key, value in results['analysis_results'].items():
            logger.info(f"{key}: {value}")

        logger.info("\n=== 模型结果 ===")
        for model_name, model_results in results['model_results'].items():
            logger.info(f"\n{model_name}:")
            # 添加类型检查
            if isinstance(model_results, dict):
                for metric, score in model_results.items():
                    if isinstance(score, (int, float)):
                        logger.info(f"{metric}: {score:.4f}")
                    else:
                        logger.info(f"{metric}: {score}")
            else:
                logger.info(f"结果: {model_results}")

        logger.info("\nPipeline执行完成")

    except Exception as e:
        logger.error(f"执行失败: {str(e)}", exc_info=True)
        raise
