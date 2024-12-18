import numpy as np
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from sklearn.base import BaseEstimator, ClassifierMixin


class ModelEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.models = {
            'lstm': None,
            'xgboost': None,
            'lightgbm': None,
            'catboost': None
        }
        self.ensemble = None
        self.weights = None

    def build_ensemble(self):
        """构建集成模型"""
        # 初始化基础模型
        base_models = []

        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        base_models.append(('xgb', xgb_model))

        # LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            verbose=-1
        )
        base_models.append(('lgb', lgb_model))

        # CatBoost
        cat_model = CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            depth=6,
            verbose=False,
            allow_writing_files=False
        )
        base_models.append(('cat', cat_model))

        # 创建投票分类器
        self.ensemble = VotingClassifier(
            estimators=base_models,
            voting='soft',
            weights=self.weights if self.weights else None,
            n_jobs=-1
        )

        # 保存模型引用
        self.models = dict(base_models)

    def fit(self, X, y):
        """实现fit方法"""
        self.ensemble.fit(X, y)
        return self

    def predict(self, X):
        """实现predict方法"""
        return self.ensemble.predict(X)

    def predict_proba(self, X):
        """实现predict_proba方法"""
        return self.ensemble.predict_proba(X)

    def optimize_weights(self, X_val, y_val):
        """优化集成权重"""
        best_score = 0
        best_weights = None

        # 网格搜索最优权重
        weight_options = np.arange(0.1, 1.0, 0.1)
        for w1 in weight_options:
            for w2 in weight_options:
                for w3 in weight_options:
                    if np.isclose(w1 + w2 + w3, 1.0):
                        weights = [w1, w2, w3]
                        self.ensemble.weights = weights
                        score = self.ensemble.score(X_val, y_val)

                        if score > best_score:
                            best_score = score
                            best_weights = weights

        self.weights = best_weights
        return best_weights
