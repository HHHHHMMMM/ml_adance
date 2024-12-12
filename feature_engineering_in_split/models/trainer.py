
import pandas as pd
from typing import Optional, Union, List, Dict, Tuple

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

import logging



class ModelTrainer:
    """模型训练器"""

    def __init__(self, model_type: str, config: Dict, logger=None):
        self.model_type = model_type
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.model = self._init_model()

    def _init_model(self):
        if self.model_type == 'xgb':
            return xgb.XGBClassifier(**self.config)
        elif self.model_type == 'lgb':
            return lgb.LGBMClassifier(**self.config)
        elif self.model_type == 'catboost':
            return cb.CatBoostClassifier(**self.config)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        try:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=10,
                verbose=False
            )
            return self._evaluate(X_train, y_train, X_val, y_val)
        except Exception as e:
            self.logger.error(f"模型训练失败: {str(e)}")
            return {'error': str(e)}

    def _evaluate(self, X_train, y_train, X_val, y_val) -> Dict:
        results = {}
        for name, (X, y) in [('train', (X_train, y_train)), ('val', (X_val, y_val))]:
            y_pred = self.model.predict(X)
            results[f'{name}_accuracy'] = accuracy_score(y, y_pred)
            results[f'{name}_f1'] = f1_score(y, y_pred, average='weighted')
        return results