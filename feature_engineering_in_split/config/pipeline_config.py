from dataclasses import dataclass, field
from typing import Dict, List, Optional

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
    min_samples_required: int = 10000

@dataclass
class TraditionalMLConfig(ModelConfig):
    """传统机器学习模型配置"""
    model_type: str = 'rf'
    cv_folds: int = 5
    max_samples_for_cv: int = 100000

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